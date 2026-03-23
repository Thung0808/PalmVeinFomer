from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x * 100:.{digits}f}%"


def _fmt_float(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _print_table(rows: list[dict], headers: list[tuple[str, str]]) -> None:
    if not rows:
        print("(no rows)")
        return

    widths: list[int] = []
    for key, title in headers:
        max_len = len(title)
        for r in rows:
            max_len = max(max_len, len(str(r.get(key, ""))))
        widths.append(max_len)

    sep = " | "
    line = "-+-".join("-" * w for w in widths)
    print(sep.join(title.ljust(widths[i]) for i, (_, title) in enumerate(headers)))
    print(line)
    for r in rows:
        print(sep.join(str(r.get(key, "")).ljust(widths[i]) for i, (key, _) in enumerate(headers)))


def main() -> None:
    p = argparse.ArgumentParser(description="Print compact horizontal table for a PVTree run.")
    p.add_argument("--run-dir", type=str, required=True, help="Path to run output dir (contains history.json).")
    p.add_argument("--topk", type=int, default=10, help="Rows to print from history.")
    p.add_argument(
        "--sort-by",
        type=str,
        default="epoch",
        choices=["epoch", "val_eer", "val_tar_1e4", "val_tar_1e5"],
        help="Sort criterion for history table.",
    )
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    history = _load_json(run_dir / "history.json") or []
    test_metrics = _load_json(run_dir / "test_metrics.json") or {}
    best_by_eer = _load_json(run_dir / "best_epochs_by_eer.json") or []

    if history:
        if args.sort_by == "epoch":
            history = sorted(history, key=lambda x: int(x.get("epoch", 0)))
        elif args.sort_by == "val_eer":
            history = sorted(history, key=lambda x: float(x.get("val_eer", 1e9)))
        elif args.sort_by == "val_tar_1e4":
            history = sorted(history, key=lambda x: float(x.get("val_tar_1e4", -1e9)), reverse=True)
        else:
            history = sorted(history, key=lambda x: float(x.get("val_tar_1e5", -1e9)), reverse=True)

        rows = []
        for r in history[: max(1, args.topk)]:
            rows.append(
                {
                    "epoch": r.get("epoch", "-"),
                    "train_acc": _fmt_pct(r.get("train_acc")),
                    "train_top5": _fmt_pct(r.get("train_top5")),
                    "val_eer": _fmt_pct(r.get("val_eer")),
                    "val_tar1e4": _fmt_pct(r.get("val_tar_1e4")),
                    "val_tar1e5": _fmt_pct(r.get("val_tar_1e5")),
                    "val_auc": _fmt_float(r.get("val_auc")),
                    "lr": _fmt_float(r.get("lr"), digits=6),
                }
            )

        print(f"Run: {run_dir}")
        print(f"History rows shown: {len(rows)} (sort={args.sort_by})")
        _print_table(
            rows,
            [
                ("epoch", "epoch"),
                ("train_acc", "train_acc"),
                ("train_top5", "train_top5"),
                ("val_eer", "val_eer"),
                ("val_tar1e4", "val_tar@1e-4"),
                ("val_tar1e5", "val_tar@1e-5"),
                ("val_auc", "val_auc"),
                ("lr", "lr"),
            ],
        )
        print()

    if best_by_eer:
        rows = []
        for r in best_by_eer[: max(1, args.topk)]:
            rows.append(
                {
                    "epoch": r.get("epoch", "-"),
                    "val_eer": _fmt_pct(r.get("val_eer")),
                    "val_tar1e4": _fmt_pct(r.get("val_tar_1e4")),
                    "val_tar1e5": _fmt_pct(r.get("val_tar_1e5")),
                    "val_auc": _fmt_float(r.get("val_auc")),
                }
            )
        print("Best epochs by EER:")
        _print_table(
            rows,
            [
                ("epoch", "epoch"),
                ("val_eer", "val_eer"),
                ("val_tar1e4", "val_tar@1e-4"),
                ("val_tar1e5", "val_tar@1e-5"),
                ("val_auc", "val_auc"),
            ],
        )
        print()

    if test_metrics:
        if "test" in test_metrics and isinstance(test_metrics["test"], dict):
            test = test_metrics["test"]
        else:
            test = {
                "eer": test_metrics.get("eer"),
                "tar_at_far_1e4": test_metrics.get("tar_at_far_1e4"),
                "tar_at_far_1e5": test_metrics.get("tar_at_far_1e5"),
                "auc": test_metrics.get("auc"),
            }
        print("Test summary:")
        print(f"best_epoch      : {test_metrics.get('best_epoch', '-')}")
        print(f"best_val_eer    : {_fmt_pct(test_metrics.get('best_val_eer'))}")
        print(f"best_val_tar1e4 : {_fmt_pct(test_metrics.get('best_val_tar_1e4'))}")
        print(f"best_val_tar1e5 : {_fmt_pct(test_metrics.get('best_val_tar_1e5'))}")
        print(f"test_eer        : {_fmt_pct(test.get('eer'))}")
        print(f"test_tar1e4     : {_fmt_pct(test.get('tar_at_far_1e4'))}")
        print(f"test_tar1e5     : {_fmt_pct(test.get('tar_at_far_1e5'))}")
        print(f"test_auc        : {_fmt_float(test.get('auc'))}")


if __name__ == "__main__":
    main()
