# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""数据读取与列角色解析。

职责边界：只负责把「用户手上的表格」变成干净的 (X, y, groups, batch, ...)，
不做任何统计、不训练模型。所有会污染特征矩阵的角色列（id/batch/replicate/target）
在这里就被剥离，下游模块无从泄漏。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ConfigError
from .utils import as_path, info, non_ascii, warn

__all__ = ["Dataset", "load_dataset", "read_table", "ID_NAME_HINTS",
           "impute_for_diagnostics"]

ID_NAME_HINTS = [
    "sample_id", "sampleid", "sample", "specimen_id", "specimen", "id",
    "样品编号", "样本编号", "样品号", "样本号", "编号", "序号",
]


@dataclass
class Dataset:
    """一次建模所需的全部结构化输入。"""

    X: np.ndarray                     # (n, p) float32
    y: np.ndarray                     # (n,) float32 或 int
    groups: np.ndarray                # (n,) str —— 分组键（sample_id）
    feature_names: list[str]
    row_ids: np.ndarray               # (n,) 行级唯一标识（用于 OOF 对齐）
    batch: np.ndarray | None = None   # (n,) str
    replicate: np.ndarray | None = None
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_name: str = "y"
    # 目标变量单位（"%"、"mg/kg"…）。None 表示作者没填——展示层会显式写"单位未指定"，
    # 不会静默省略；空字符串表示明确无量纲（如 RPD、斜率）。
    target_unit: str | None = None
    target_resolution: str | float | None = None
    task: str = "regression"
    classes: list[str] | None = None
    positive_label: Any = None
    notes: list[str] = field(default_factory=list)
    # 因 dtype 非数值而被丢弃的列（未 one-hot，也未出现在 feature_cols 里）。
    # German Credit 这类「类别列以字符串/整数编码混在表里」的数据会触发；
    # 审计必须把它升级成 warn，否则模型只用了 7/20 列而报告看上去一切正常。
    dropped_non_numeric: list[str] = field(default_factory=list)
    source_path: str | None = None
    data_hash: str | None = None
    wavelength: np.ndarray | None = None
    label_map: dict[int, Any] = field(default_factory=dict)
    targets: dict[str, np.ndarray] = field(default_factory=dict)
    # 多目标行对齐的审计信息：每个目标「原表有效行数 → 实际保留行数」以及
    # 因兄弟目标缺行而额外损失的行数。只用于报告，不参与建模。
    target_available: dict[str, int] = field(default_factory=dict)
    target_lost: dict[str, int] = field(default_factory=dict)
    dropped_all_targets: int = 0

    @property
    def target_names(self) -> list[str]:
        return list(self.targets) if self.targets else [self.target_name]

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_groups(self) -> int:
        return int(len(pd.unique(self.groups)))

    def groups_per_sample(self) -> pd.Series:
        return pd.Series(self.groups).value_counts()

    def subset(self, mask: np.ndarray) -> "Dataset":
        mask = np.asarray(mask, dtype=bool)
        return Dataset(
            X=self.X[mask],
            y=self.y[mask],
            groups=self.groups[mask],
            feature_names=list(self.feature_names),
            row_ids=self.row_ids[mask],
            batch=None if self.batch is None else self.batch[mask],
            replicate=None if self.replicate is None else self.replicate[mask],
            meta=self.meta.loc[mask].reset_index(drop=True),
            target_name=self.target_name,
            target_unit=self.target_unit,
            target_resolution=self.target_resolution,
            task=self.task,
            classes=self.classes,
            positive_label=self.positive_label,
            notes=list(self.notes),
            source_path=self.source_path,
            data_hash=self.data_hash,
            wavelength=self.wavelength,
            label_map=dict(self.label_map),
            targets={k: np.asarray(v)[mask] for k, v in self.targets.items()},
            target_available=dict(self.target_available),
            target_lost=dict(self.target_lost),
            dropped_all_targets=self.dropped_all_targets,
        )


def resolve_target_field(cfg: dict, key: str, target: str | None = None):
    """读取 ``data.<key>``，支持标量或 ``{目标列名: 值}`` 两种写法。

    **字典缺 key 时返回 None，绝不回退到别的目标的单位。** 回退是最省事也最危险的做法：
    多目标 run 里一个目标以 % 计、另一个以 mg/kg 计，混用会把 RMSE 标错量纲，
    而且全程不报任何错——读者只会以为这个模型精度高得离谱。
    """
    raw = (cfg.get("data") or {}).get(key)
    if raw is None:
        return None
    if isinstance(raw, dict):
        if target is None:
            return None
        val = raw.get(str(target))
        return None if val is None else val
    return raw


# --------------------------------------------------------------------------- #
# 读取原始表
# --------------------------------------------------------------------------- #
def _read_cache_path(p: Path, sheet, encoding: str, sep: str | None) -> Path | None:
    """按「文件内容哈希 + 读取参数」定位解析缓存，避免每个 stage 重复解析同一份 Excel。"""
    import os
    import tempfile

    from .utils import sha256_file

    try:
        stamp = sha256_file(p)
    except OSError:
        return None
    base = Path(os.environ.get("SML_CACHE_DIR") or (Path(tempfile.gettempdir()) / "sml-dfcache"))
    key = hashlib.sha256(
        f"{p}|{stamp}|{sheet}|{encoding}|{sep}|{pd.__version__}".encode("utf-8")
    ).hexdigest()[:32]
    return base / f"{key}.pkl"


def read_table_cached(
    path: str | Path,
    sheet: int | str = 0,
    encoding: str = "utf-8",
    sep: str | None = None,
) -> pd.DataFrame:
    """``read_table`` 的磁盘缓存版：内容哈希命中时直接读 pickle。

    只缓存**原始解析结果**（不含任何按配置做的列角色处理），因此后续的重命名、
    na_values 替换、长转宽都照常执行，结果与不缓存时完全一致。文件内容一变，
    sha256 变、键变、缓存自然失效。
    """
    p = as_path(path)
    cache = _read_cache_path(p, sheet, encoding, sep) if p.exists() else None
    if cache is not None and cache.exists():
        try:
            df = pd.read_pickle(cache)
            info(f"数据解析缓存命中：{p.name}（{df.shape[0]} 行 × {df.shape[1]} 列）")
            return df
        except Exception:  # noqa: BLE001 —— 缓存损坏就当没有
            pass
    df = read_table(p, sheet=sheet, encoding=encoding, sep=sep)
    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(cache)
        except Exception:  # noqa: BLE001 —— 写不进缓存不影响主流程
            pass
    return df


def read_table(
    path: str | Path,
    sheet: int | str = 0,
    encoding: str = "utf-8",
    sep: str | None = None,
) -> pd.DataFrame:
    p = as_path(path)
    if not p.exists():
        raise ConfigError("E-IO-001", f"数据文件不存在：{p}")
    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        try:
            df = pd.read_excel(p, sheet_name=sheet)
        except ImportError as exc:  # 缺 openpyxl / xlrd
            raise ConfigError("E-IO-002", f"读取 Excel 失败，缺少依赖：{exc}") from exc
    elif suffix in (".csv", ".txt", ".tsv", ".dat"):
        kwargs: dict[str, Any] = {"encoding": encoding}
        if sep is not None:
            kwargs["sep"] = sep
        else:
            kwargs["sep"] = None
            kwargs["engine"] = "python"
        try:
            df = pd.read_csv(p, **kwargs)
        except UnicodeDecodeError:
            info(f"{p.name} 非 utf-8 编码，回退 gb18030")
            kwargs["encoding"] = "gb18030"
            df = pd.read_csv(p, **kwargs)
    elif suffix in (".parquet",):
        df = pd.read_parquet(p)
    else:
        raise ConfigError("E-IO-003", f"不支持的数据格式：{suffix}")
    if df.empty:
        raise ConfigError("E-IO-004", f"数据文件为空：{p}")
    # 全空列名（Excel 常见）重命名，避免 "Unnamed: 12"；
    # 同时剥掉 UTF-8 BOM：Excel 另存的 CSV 会把 \ufeff 粘在首列表头上，
    # 否则 data.id_col 永远匹配不上（报 E-IO-015 却不提 BOM），
    # 留空 id_col 时会静默退化到指纹分组，破坏「同一样品」的分组语义。
    df = df.rename(columns={c: str(c).replace("\ufeff", "").strip() for c in df.columns})
    return df


# --------------------------------------------------------------------------- #
# 列角色解析
# --------------------------------------------------------------------------- #
def _guess_id_col(df: pd.DataFrame) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for hint in ID_NAME_HINTS:
        if hint.lower() in lowered:
            return lowered[hint.lower()]
    return None


def _fingerprint_ids(X: np.ndarray) -> np.ndarray:
    """特征指纹兜底：同一行数值完全相同 -> 同一 group。"""
    rounded = np.round(np.asarray(X, dtype=np.float64), 4)
    hashes = []
    for row in rounded:
        payload = ",".join(f"{v:.4f}" for v in row).encode("utf-8")
        hashes.append(hashlib.md5(payload).hexdigest()[:12])
    uniq = {h: f"auto_{i + 1:04d}" for i, h in enumerate(dict.fromkeys(hashes))}
    return np.array([uniq[h] for h in hashes], dtype=object)


def _coerce_long(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = cfg["data"]["long_cols"]
    id_col, xcol, ycol = cols.get("id"), cols.get("x"), cols.get("y")
    if not all([id_col, xcol, ycol]):
        raise ConfigError("E-IO-005", "data.format=long 时必须给出 long_cols.{id,x,y}")
    missing = [c for c in (id_col, xcol, ycol) if c not in df.columns]
    if missing:
        raise ConfigError("E-IO-006", f"长表缺少列：{missing}")
    wide = df.pivot_table(index=id_col, columns=xcol, values=ycol, aggfunc="mean")
    wide = wide.sort_index(axis=1)
    if wide.isna().any().any():
        n_missing = int(wide.isna().sum().sum())
        raise ConfigError(
            "E-IO-007",
            f"长表转宽表后有 {n_missing} 个缺失格，说明各样品波长轴不一致；"
            f"请先统一定标或改用意插值。当前拒绝静默对齐。",
        )
    wide = wide.reset_index()
    info(f"长表转宽表：{wide.shape[0]} 样品 × {wide.shape[1] - 1} 波数点")
    return wide


def _wavelength_axis(cfg: dict, n_features: int) -> np.ndarray | None:
    axis = cfg["data"].get("wavelength_axis")
    if not axis:
        return None
    start = float(axis.get("start"))
    stop = float(axis.get("stop"))
    step = float(axis.get("step"))
    if step == 0:
        raise ConfigError("E-IO-008", "wavelength_axis.step 不能为 0")
    n_expect = int(round(abs(stop - start) / abs(step))) + 1
    if n_expect != n_features:
        warn(
            f"波数轴推算得到 {n_expect} 点，但实际特征数 {n_features}；"
            f"按线性等距重建 {n_features} 点（画图用，不影响建模）"
        )
    return np.linspace(start, stop, n_features)


def _encode_target(raw: pd.Series, cfg: dict) -> tuple[np.ndarray, str, list[str] | None, Any, dict]:
    task = cfg["task"]["type"]
    if task == "regression":
        y = pd.to_numeric(raw, errors="coerce")
        # 空值是允许的（多目标时该行仍可给其它目标用）；但非空又解析不了的文本必须报错，
        # 否则会被静默当成缺失，最后只剩「有效行变少」一个模糊现象。
        blank = raw.isna() | raw.astype(str).str.strip().eq("")
        bad = y.isna() & ~blank
        if bad.any():
            vals = list(pd.unique(raw[bad].astype(str)))[:5]
            raise ConfigError(
                "E-IO-021",
                f"目标列含无法解析为数值的值：{vals}"
                + ("…" if bad.sum() > len(vals) else ""),
            )
        return y.to_numpy(dtype=np.float64), "regression", None, None, {}

    order = cfg["task"].get("label_order")
    labels = list(pd.unique(raw.astype(str)))
    if order:
        missing = [v for v in labels if v not in [str(o) for o in order]]
        if missing:
            raise ConfigError("E-IO-009", f"task.label_order 未覆盖数据中的标签：{missing}")
        classes = [str(o) for o in order]
    else:
        classes = sorted(labels)
    if len(classes) < 2:
        raise ConfigError("E-IO-010", f"分类任务至少需要 2 个类别，当前只有 {classes}")
    pos = cfg["task"].get("positive_label")
    if pos is None:
        pos = classes[-1]
    pos = str(pos)
    if pos not in classes:
        raise ConfigError("E-IO-011", f"positive_label={pos!r} 不在类别 {classes} 中")
    mapping = {c: i for i, c in enumerate(classes)}
    y = raw.astype(str).map(mapping).to_numpy()
    if pd.isna(y).any():
        raise ConfigError("E-IO-012", "标签编码后出现缺失，检查原始标签的空值")
    label_map = {i: c for c, i in mapping.items()}
    return y.astype(np.int64), "classification", classes, pos, label_map


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def load_dataset(cfg: dict, path: str | Path | None = None) -> tuple[Dataset, pd.DataFrame]:
    """按配置读取数据，返回 (Dataset, 原始宽表)。

    第二个返回值用于审计阶段统计原始列的缺失率等信息。
    """
    data_cfg = cfg["data"]
    target_path = path or data_cfg.get("path")
    if not target_path:
        raise ConfigError("E-IO-013", "未指定数据文件（--data 或 data.path）")
    p = as_path(target_path)

    df = read_table_cached(p, sheet=data_cfg.get("sheet", 0),
                           encoding=data_cfg.get("encoding", "utf-8"),
                           sep=data_cfg.get("sep"))
    notes: list[str] = []
    if data_cfg.get("format") == "long":
        df = _coerce_long(df, cfg)

    na_values = data_cfg.get("na_values") or []
    if na_values:
        df = df.replace({v: np.nan for v in na_values})

    target_col = cfg["task"]["target"][0]
    missing_targets = [t for t in cfg["task"]["target"] if str(t) not in df.columns]
    if missing_targets:
        cols_preview = list(df.columns)[:15]
        raise ConfigError(
            "E-IO-014",
            f"目标列 {missing_targets} 不在表头中。当前列（前 15 个）：{cols_preview}",
        )

    # ---- 角色列 ----
    id_col = data_cfg.get("id_col") or _guess_id_col(df)
    if id_col and id_col not in df.columns:
        raise ConfigError("E-IO-015", f"data.id_col={id_col!r} 不在表头中")
    batch_col = data_cfg.get("batch_col")
    if batch_col and batch_col not in df.columns:
        raise ConfigError("E-IO-016", f"data.batch_col={batch_col!r} 不在表头中")
    rep_col = data_cfg.get("replicate_col")
    if rep_col and rep_col not in df.columns:
        raise ConfigError("E-IO-017", f"data.replicate_col={rep_col!r} 不在表头中")

    # split.group_col 曾经是死字段：只在报告/审计里拼显示字符串，真正的分组键永远是
    # data.id_col。用户写了 group_col 却看到 CV 按 id_col 分组，且没有任何提示。
    # 现在让它真正生效：显式给了就用它，没给才回退 id_col。
    group_col = cfg.get("split", {}).get("group_col") or None
    if group_col and group_col not in df.columns:
        raise ConfigError(
            "E-IO-024",
            f"split.group_col={group_col!r} 不在表头中。若想用 sample_id 分组请写 "
            f"split.group_col=sample_id；留空则回退 data.id_col",
        )
    if group_col and id_col and group_col != id_col:
        notes.append(f"分组键取 split.group_col={group_col!r}（而非 data.id_col={id_col!r}）")

    # 分组键必须从特征里剔除：它是样品标识，进特征等于把"样品身份"当预测依据。
    roles = {c for c in (id_col, batch_col, rep_col, target_col, group_col) if c}
    roles.update(str(t) for t in cfg["task"]["target"])      # 全部目标列都不是特征
    roles.update(data_cfg.get("drop_cols") or [])

    # 显式排除（fan-out 拆子 run 时用来传入"兄弟目标列"，见 run_pipeline）
    excluded = [str(c) for c in (data_cfg.get("exclude_from_features") or [])]
    if excluded:
        missing_excl = [c for c in excluded if c not in df.columns]
        if missing_excl:
            raise ConfigError(
                "E-IO-023",
                f"data.exclude_from_features 中列不存在：{missing_excl[:8]}。"
                f"当前列（前 15 个）：{list(df.columns)[:15]}",
            )
        roles.update(excluded)

    # 多目标：兄弟目标列必须从特征里剔除。用"蛋白"列去预测"水分"会白拿高分，
    # 而审计的通用泄漏检查（单特征完美预测）抓不到——它看起来就是个普通强相关特征。
    all_targets = [str(t) for t in cfg["task"]["target"]]
    if len(all_targets) > 1:
        notes.append(f"多目标建模：{all_targets} 均已从特征矩阵中剔除"
                     f"（否则会用兄弟目标列预测当前目标，分数虚高且审计难以识别）")
    extra_excl = [c for c in excluded if c not in all_targets]
    if extra_excl:
        notes.append(f"按 data.exclude_from_features 从特征中排除：{extra_excl}")

    # ---- 特征列 ----
    # data.feature_mode 之前只在 DEFAULTS 里定义、从来没有被读过（改它没有任何效果）。
    # 现在只有一种实现方式，就把它变成一个"取值必须对得上"的字段：
    # 写错值直接报错，而不是让它安静地什么都不做。
    mode = str(data_cfg.get("feature_mode") or "all_numeric_except_roles")
    if mode != "all_numeric_except_roles":
        raise ConfigError(
            "E-IO-025",
            f"data.feature_mode 目前只实现了一种语义 all_numeric_except_roles（所有数值列减去"
            f"角色列），收到 {mode!r}。\n"
            f"  要精确控制特征，请用 data.feature_cols 显式列出（此时 feature_mode 不生效），"
            f"或用 data.drop_cols 排除。",
        )

    explicit = data_cfg.get("feature_cols")
    non_numeric_dropped: list[str] = []
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise ConfigError("E-IO-018", f"data.feature_cols 中列不存在：{missing[:8]}")
        feature_cols = [c for c in explicit if c not in roles]
    else:
        feature_cols = [
            c for c in df.columns
            if c not in roles and pd.api.types.is_numeric_dtype(df[c])
        ]
        non_numeric_dropped = [
            c for c in df.columns
            if c not in roles and not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric_dropped:
            notes.append(
                f"以下非数值列未作为特征（若本应是类别特征，请先独热编码或指定 feature_cols）："
                f"{non_numeric_dropped[:10]}"
            )
    if not feature_cols:
        raise ConfigError("E-IO-019", "没有解析到任何数值特征列；检查 feature_cols / drop_cols")

    # ---- 缺失去向：丢弃缺失过多的特征 ----
    min_valid = float(data_cfg.get("min_valid_frac", 0.5))
    Xdf = df[feature_cols].astype(np.float64)
    valid_frac = Xdf.notna().mean(axis=0)
    dropped_na = [c for c in feature_cols if valid_frac[c] < min_valid]
    if dropped_na:
        notes.append(f"非缺失比例 <{min_valid:.0%} 的 {len(dropped_na)} 个特征被丢弃：{dropped_na[:8]}")
        feature_cols = [c for c in feature_cols if c not in dropped_na]
        Xdf = Xdf[feature_cols]
    if not feature_cols:
        raise ConfigError("E-IO-020", "所有特征列缺失率都过高，已无可用特征")

    # ---- 行级缺失：只丢「所有目标都缺」的行 ----
    # 多目标时各目标共用同一份 X，但**不能**因为目标 A 缺一行就把该行从 B 里也删掉。
    # 这里保留「至少一个目标有效」的行；CV 阶段会按目标各自的有限值掩码再取子集。
    # 同时记录每个目标「原表有效行数 → 实际保留行数」，审计阶段用 warn 级别明确展示，
    # 避免出现「CSV 15 行、dry-run 14 行」却只在载入说明里写一句的隐蔽损失。
    tnames = [str(t) for t in cfg["task"]["target"]]
    miss = {t: df[t].isna().to_numpy() for t in tnames}
    keep = np.zeros(len(df), dtype=bool)
    for t in tnames:
        keep |= ~miss[t]
    n_drop_all = int((~keep).sum())
    bad_rows = (np.flatnonzero(~keep) + 1).tolist()
    shown = bad_rows[:5]
    row_txt = "、".join(str(r) for r in shown) + ("…" if len(bad_rows) > len(shown) else "")
    avail = {t: int(((~miss[t]) & keep).sum()) for t in tnames}
    lost = {t: int((~miss[t]).sum()) - avail[t] for t in tnames}
    if len(tnames) == 1:
        if n_drop_all:
            notes.append(f"目标 {tnames[0]} 缺 {n_drop_all} 行（原表第 {row_txt} 行），已剔除")
    else:
        if n_drop_all:
            notes.append(
                f"所有目标均缺失的 {n_drop_all} 行已剔除（原表第 {row_txt} 行）；"
                f"其余行按各目标自身有效值分别进入建模"
            )
        lost_targets = [t for t in tnames if lost[t] > 0]
        if lost_targets:
            tail = "；".join(
                f"{t} 有效 {int((~miss[t]).sum())} → 保留 {avail[t]}"
                + (f"（因兄弟目标缺行损失 {lost[t]}）" if lost[t] else "")
                for t in tnames)
            warn(
                f"多目标行对齐：{'、'.join(lost_targets)} 缺行，"
                f"这些行仍保留给其它目标（不再整行连带剔除）。{tail}。"
                f"若某目标对样本量敏感，建议单独跑它：--set task.target=[\"{tnames[0]}\"]"
            )
    df = df.loc[keep].reset_index(drop=True)
    Xdf = Xdf.loc[keep].reset_index(drop=True)

    # 特征缺失保留到 Pipeline：中位数必须在每个训练折内部估计。
    # 旧实现在载入阶段用全量数据的中位数填补，虽然只影响少数缺失行，但严格来说
    # 已经让测试折的分布信息进入了训练前的预处理（轻量泄漏），与技能自己的
    # "fit-time 统计量只看训练折"原则冲突。这里只记录，不填。
    na_rows = int(Xdf.isna().any(axis=1).sum())
    if na_rows:
        notes.append(
            f"{na_rows} 行存在特征缺失；缺失值保留到 Pipeline 的 SimpleImputer"
            f"（中位数只在训练折内估计）。审计/AD 等全量诊断会另行用列中位数插补并标注。")
    X = Xdf.to_numpy(dtype=np.float32)

    # ---- 分组 ----
    key_col = group_col or id_col
    if key_col:
        groups = df[key_col].astype(str).to_numpy()
        n_unique = len(pd.unique(groups))
        if n_unique == len(groups):
            notes.append(
                f"{key_col} 每行都唯一，说明数据里没有重复测量；分组 CV 退化为普通 K 折 "
                f"（若同一材料确实多次测量但未标注，请补 replicate/sample_id）"
            )
    else:
        groups = _fingerprint_ids(X)
        notes.append(
            "未提供 sample_id，已按特征指纹自动推断分组（数值完全相同的行视为同一样品）。"
            "若同一样品的多次测量存在明显谱图差异，该推断会失效——请务必补充 sample_id 列。"
        )

    batch = df[batch_col].astype(str).to_numpy() if batch_col else None
    replicate = df[rep_col].to_numpy() if rep_col else None

    # ---- 目标（支持多目标回归；多目标分类无意义，直接拒绝） ----
    target_names = [str(t) for t in cfg["task"]["target"]]
    if cfg["task"]["type"] == "classification" and len(target_names) > 1:
        raise ConfigError("E-IO-022", "分类任务只支持单个目标列；多标签请先自行处理")
    targets_dict: dict[str, np.ndarray] = {}
    task, classes, pos, label_map = "regression", None, None, {}
    for tname in target_names:
        y_t, task, classes, pos, label_map = _encode_target(df[tname], cfg)
        targets_dict[tname] = y_t
    y = targets_dict[target_col]

    wavelength = _wavelength_axis(cfg, len(feature_cols))

    meta_cols = [c for c in (id_col, batch_col, rep_col) if c]
    meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)
    meta.insert(0, "row_id", np.arange(len(df)))

    row_ids = np.array([f"r{i:06d}" for i in range(len(df))], dtype=object)

    # ---- 数据指纹（复现已锁） ----
    fp_df = df.copy()
    if id_col:
        fp_df = fp_df.sort_values(by=[id_col], kind="stable")
    payload = fp_df.to_csv(index=False).encode("utf-8")
    data_hash = hashlib.sha256(payload).hexdigest()

    ds = Dataset(
        X=X, y=y, groups=groups.astype(object),
        feature_names=list(feature_cols), row_ids=row_ids,
        batch=batch, replicate=replicate, meta=meta,
        target_name=target_col, target_unit=resolve_target_field(cfg, "target_unit", target_col),
        target_resolution=resolve_target_field(cfg, "target_resolution", target_col),
        task=task, classes=classes,
        positive_label=pos, notes=notes, source_path=str(p), data_hash=data_hash,
        dropped_non_numeric=list(non_numeric_dropped),
        wavelength=wavelength, label_map=dict(label_map), targets=targets_dict,
        target_available=avail, target_lost=lost, dropped_all_targets=n_drop_all,
    )

    info(
        f"数据载入：{ds.n_samples} 行 × {ds.n_features} 特征 | "
        f"{ds.n_groups} 个 group | 任务={task} | 目标={','.join(target_names)}"
        + (f" | 批次 {len(pd.unique(batch))} 个" if batch is not None else "")
    )
    if non_ascii(str(p)):
        notes.append(
            f"数据路径含非 ASCII 字符（{p.parent}），若使用多进程调参出现 pickle 报错，"
            f"请把数据与 --out 都放到英文路径下"
        )
    return ds, df


def impute_for_diagnostics(X: np.ndarray) -> np.ndarray:
    """全量诊断（AD/VIP/SHAP/画图）用的中位数插补。

    注意：这里故意用全量中位数，因为 AD/VIP/SHAP 本身就是在全量数据上做的
    描述性分析，不是交叉验证性能估计；CV 内的缺失值由 Pipeline 的
    ``SimpleImputer`` 在每个训练折内单独拟合，两者不要混用。
    """
    A = np.asarray(X, dtype=np.float64).copy()
    if not np.isnan(A).any():
        return A.astype(np.float32)
    med = np.nanmedian(A, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(~np.isfinite(A))
    A[inds] = med[inds[1]]
    return A.astype(np.float32)


def load_from_run(cfg: dict) -> Dataset:
    return load_dataset(cfg)[0]
