"""用已完成的 run 预测新样品（含适用域判定）。

典型用法
--------
    # 用 CV 选出的最优模型 + 最常用超参，在全量数据上重训并保存
    python predict.py --run-dir D:/runs/nir_v1/targets/01_水分 --retrain

    # 预测一张新样品表（列名必须与训练特征一致）
    python predict.py --run-dir D:/runs/nir_v1/targets/01_水分 --data D:/data/new.csv

    # 不给 --out 时写到 <run>/predictions/predictions.csv；--no-ad 跳过适用域
    python predict.py --run-dir ... --data new.xlsx --out D:/pred/new_pred.csv

工艺窗口的做法：把候选工艺参数/光谱组合写成 CSV（一行一个待预测样品，列名与训练
特征一致），再用本脚本批量预测；`ad_flag=1` 的行表示落在适用域之外，不要直接外推。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sml.config import ConfigError  # noqa: E402
from sml.final_model import predict_frame, save_final_model  # noqa: E402
from sml.io_data import load_dataset, read_table  # noqa: E402
from sml.stage_common import RunContext  # noqa: E402
from sml.utils import (  # noqa: E402
    info, quiet_library_warnings, read_json, set_verbose, setup_console,
)


def _retrain(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.resolved.json"
    if not cfg_path.exists():
        raise ConfigError(
            "E-PRED-004",
            f"{run_dir} 下没有 config.resolved.json，无法重建最终模型。"
            f"请先跑完整 run（至少 stage_cv），或用 --run-dir 指向正确的子 run。",
        )
    cfg = read_json(cfg_path)
    ds, _ = load_dataset(cfg)
    return save_final_model(RunContext(cfg, ds, run_dir))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="用已完成的 run 预测新样品（可选适用域判定）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-dir", required=True, help="已完成 stage_cv 的 run 目录")
    ap.add_argument("--data", default=None, help="新样品表（CSV/Excel），列名须与训练特征一致")
    ap.add_argument("--out", default=None,
                    help="输出 CSV 路径或目录；默认 <run>/predictions/predictions.csv")
    ap.add_argument("--no-ad", action="store_true", help="跳过适用域判定")
    ap.add_argument("--retrain", action="store_true",
                    help="用 CV 最优模型 + 最常用超参在全量数据上重建最终模型")
    ap.add_argument("--quiet", action="store_true", help="减少控制台输出")
    args = ap.parse_args(argv)

    setup_console()
    set_verbose(not args.quiet)
    if args.quiet:
        quiet_library_warnings()

    run_dir = Path(args.run_dir).expanduser().resolve()
    model_path = run_dir / "model" / "final_model.joblib"
    if args.retrain or not model_path.exists():
        if not model_path.exists() and not args.retrain:
            info("未找到 model/final_model.joblib，自动从 CV 产物重建最终模型……")
        _retrain(run_dir)

    if not args.data:
        if args.retrain:
            info("未提供 --data，仅完成最终模型重建。")
            return 0
        raise ConfigError("E-PRED-003", "需要 --data <新样品表>，或加 --retrain 只重建模型")

    df = read_table(args.data)
    out_df = predict_frame(run_dir, df, use_ad=not args.no_ad)

    if args.out:
        out = Path(args.out).expanduser()
        if out.suffix.lower() in (".csv", ".txt", ".tsv"):
            out.parent.mkdir(parents=True, exist_ok=True)
            out_path = out
        else:
            out.mkdir(parents=True, exist_ok=True)
            out_path = out / "predictions.csv"
    else:
        out_dir = run_dir / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "predictions.csv"

    out_df.to_csv(out_path, index=False, encoding="utf-8")
    info(f"预测完成：{out_path}（{len(out_df)} 行）")
    if "ad_flag" in out_df.columns:
        n_flag = int((out_df["ad_flag"].to_numpy() == 1).sum())
        info(f"适用域：{n_flag}/{len(out_df)} 行被判为域外（ad_flag=1）；"
             f"这些行的预测只能作参考，不要直接外推")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as exc:
        print(f"\n[预测配置错误] {exc}\n", flush=True)
        raise SystemExit(2) from None
