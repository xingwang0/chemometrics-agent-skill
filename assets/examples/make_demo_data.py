# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""生成两套合成数据，用于 selfcheck 与新手教学。

1. ``demo_nir.csv``      —— 拟真的近红外光谱 + 成分标签，含 3 次重复测量、4 个批次，
                            并故意植入「近零方差波段」与「成对共线列」，用来验证审计能否抓出来。
2. ``leakage_probe.csv`` —— 泄漏探针：谱图唯一编码样品身份，y 与身份不可平滑映射。
                            随机 KFold 会靠"记住同一样品的另一次测量"拿到高 R²；
                            分组 KFold 面对没见过的样品应立刻崩掉。
                            这是把"防泄漏"从口号变成可执行断言的回归测试数据。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WN_START = 4000
WN_STOP = 10000
WN_STEP = 8

PEAKS = {
    # 近红外区（4000–10000 cm⁻¹）真实存在的合频/倍频吸收带，不是中红外的基频峰
    "protein_amide": (4900.0, 90.0),      # 酰胺 N-H 合频
    "protein_ch": (6300.0, 120.0),        # 蛋白质 C-H 一倍频
    "moisture_oh1": (5150.0, 200.0),      # 水 O-H 合频（最强）
    "moisture_oh2": (6900.0, 180.0),      # 水 O-H 一倍频
    "fat_ch1": (5800.0, 90.0),            # 脂肪 C-H 合频
    "fat_ch2": (8260.0, 150.0),           # 脂肪 C-H 二倍频
    "starch_oh": (6100.0, 110.0),         # 淀粉 O-H 一倍频
}


def _gauss(axis: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((axis - center) / width) ** 2)


def _smooth(rng: np.random.Generator, n: int, n_points: int) -> np.ndarray:
    """用少量随机节点的三次插值造一条平滑随机曲线（模拟基线漂移）。"""
    knots = 6
    xs = np.linspace(0, 1, knots)
    ys = rng.normal(0, 1, knots)
    grid = np.linspace(0, 1, n_points)
    return np.vstack([np.interp(grid, xs, rng.permutation(ys)) for _ in range(n)])


def make_demo_nir(
    out_csv: Path,
    n_samples: int = 60,
    reps: int = 3,
    n_batches: int = 4,
    seed: int = 20261122,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    axis = np.arange(WN_START, WN_STOP + 1, WN_STEP, dtype=float)
    n_wn = axis.size

    protein = rng.uniform(8.0, 19.0, n_samples)
    moisture = rng.uniform(5.0, 25.0, n_samples)
    fat = rng.uniform(1.0, 12.0, n_samples)
    starch = rng.uniform(10.0, 40.0, n_samples)

    pa = _gauss(axis, *PEAKS["protein_amide"])
    pc = _gauss(axis, *PEAKS["protein_ch"])
    oh1 = _gauss(axis, *PEAKS["moisture_oh1"])
    oh2 = _gauss(axis, *PEAKS["moisture_oh2"])
    f1 = _gauss(axis, *PEAKS["fat_ch1"])
    f2 = _gauss(axis, *PEAKS["fat_ch2"])
    st = _gauss(axis, *PEAKS["starch_oh"])

    base_shape = 0.9 + 0.15 * np.sin(axis / 900.0)
    rows: list[dict] = []
    batch_of = rng.integers(0, n_batches, n_samples)

    for i in range(n_samples):
        # 纯吸收项（比尔-朗伯，近似线性，含轻微非线性交叉项）
        absorb = (
            0.030 * protein[i] * (pa + 0.5 * pc)
            + 0.022 * moisture[i] * (oh1 + 0.6 * oh2)
            + 0.030 * fat[i] * (f1 + 0.4 * f2)
            + 0.006 * starch[i] * st
        )
        # 批次效应：每批一个固定增益与偏移（跨批次外推的真实障碍）
        gain_b = 1.0 + 0.04 * (batch_of[i] - (n_batches - 1) / 2)
        offset_b = 0.015 * (batch_of[i] - (n_batches - 1) / 2)
        for r in range(reps):
            scatter = gain_b * (1.0 + 0.02 * rng.normal())
            baseline = 0.30 * base_shape + 0.10 * _smooth(rng, 1, n_wn)[0]
            noise = rng.normal(0, 0.015, n_wn)
            spec = baseline + scatter * (absorb + offset_b) + noise
            rec = {
                "sample_id": f"S{i + 1:03d}",
                "batch": f"B{batch_of[i] + 1}",
                "replicate": r + 1,
                "protein": round(float(protein[i]), 3),
                "moisture": round(float(moisture[i]), 3),
                "fat": round(float(fat[i]), 3),
                "starch": round(float(starch[i]), 3),
                "备注": "合成演示数据" if r == 0 else "",
            }
            for j, wn in enumerate(axis):
                rec[str(int(wn))] = float(spec[j])
            rows.append(rec)

    df = pd.DataFrame(rows)

    # 植入 1：近零方差波段（仪器盲区，随机微噪声）
    df["dead_band_1"] = rng.normal(0, 1e-9, len(df))
    df["dead_band_2"] = 0.0
    # 植入 2：成对共线（moisture 的线性复制 + 微噪声）
    df["moisture_copy"] = df["moisture"] * 2.0 - 1.0
    df["moisture_copy_noisy"] = df["moisture"] * 2.0 - 1.0 + rng.normal(0, 1e-6, len(df))
    # 植入 3：疑似 ID 特征（行序号，看起来像"序号"）
    df["序号"] = np.arange(len(df)) + 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[demo] 写出 {out_csv}  ({df.shape[0]} 行 × {df.shape[1]} 列)", flush=True)
    print(f"[demo]   样品 {n_samples}，重复 {reps} 次，批次 {n_batches}，波数点 {n_wn}", flush=True)
    return df


def make_leakage_probe(
    out_csv: Path,
    n_groups: int = 40,
    reps: int = 3,
    n_points: int = 200,
    seed: int = 7,
) -> pd.DataFrame:
    """谱图 = 该组专属随机平滑曲线；y = 该组的随机取值。两者无跨组可学的映射。"""
    rng = np.random.default_rng(seed)
    axis = np.linspace(0, 1, n_points)
    knot_x = np.linspace(0, 1, 8)

    rows = []
    for g in range(n_groups):
        signature = np.interp(axis, knot_x, rng.normal(0, 1, knot_x.size))
        y_g = rng.normal(0, 1) * 10.0          # 与 signature 无稳定关系，只能"记住"
        for r in range(reps):
            spec = signature + rng.normal(0, 0.01, n_points)   # 同组三次测量几乎相同
            rec = {"sample_id": f"G{g + 1:03d}", "batch": f"B{g % 4 + 1}", "replicate": r + 1,
                   "y": round(float(y_g), 4)}
            for j in range(n_points):
                rec[f"f{j:03d}"] = float(spec[j])
            rows.append(rec)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[probe] 写出 {out_csv}  ({df.shape[0]} 行 × {df.shape[1]} 列)", flush=True)
    print(f"[probe]   {n_groups} 组 × {reps} 次重复；随机KFold应虚高、分组KFold应崩", flush=True)
    return df


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="生成 research-ml-modeling 的演示与泄漏探针数据")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--n-samples", type=int, default=60)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args(argv)

    out = Path(args.out_dir).expanduser().resolve()
    make_demo_nir(out / "demo_nir.csv", n_samples=args.n_samples, reps=args.reps)
    make_leakage_probe(out / "leakage_probe.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
