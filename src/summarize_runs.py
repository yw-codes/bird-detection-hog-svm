"""Collect the metrics of every run folder into one Markdown table."""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs")
    p.add_argument("--split", default="val")
    return p.parse_args()


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def train_minutes(folder):
    log = read(folder / "train_log.json")
    seconds = log.get("train_seconds", 0) + log.get("reranker_seconds", 0)
    if (folder / "template_log.json").exists():
        seconds += read(folder / "template_log.json")["train_seconds"]
    return seconds / 60


def main():
    args = parse_args()
    rows = []
    for path in Path(args.runs).glob(f"*/eval_{args.split}/metrics.json"):
        folder = path.parent.parent
        m = read(path)
        small = m["by_size"]["small (<32 px)"]["max_recall"]
        rows.append((m["mAP50"], f"| {folder.name} | {m['mAP50']:.4f} | {m['best_f1']:.3f} | {m['precision']:.3f} "
                                 f"| {m['recall']:.3f} | {m['max_recall']:.3f} | {small:.3f} | {m['count_mae']:.2f} "
                                 f"| {train_minutes(folder):.1f} | {m['seconds_per_image']:.2f} |"))
    print(f"Split: {args.split}\n")
    print("| Run | mAP@0.5 | Best F1 | P | R | Max recall | Max recall (small) | Count MAE | Train (min) | s/image |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for _, row in sorted(rows, reverse=True):
        print(row)


if __name__ == "__main__":
    main()
