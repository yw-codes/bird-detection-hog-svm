"""Draw the best and worst images of an evaluation, for the failure analysis in the report.

Left panel: ground truth. Right panel: detections above the best-F1 threshold, green if they matched a
bird at IoU 0.5 and red if they did not. Works from predictions.json, so nothing is run again.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import load_split
from metrics import match


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--count", type=int, default=6, help="images per group")
    return p.parse_args()


def load_detections(path, samples):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    grouped = defaultdict(list)
    for row in rows:
        x, y, w, h = row["bbox"]
        grouped[row["image_id"]].append([x, y, x + w, y + h, row["score"]])
    detections = {}
    for sample in samples:
        array = np.array(grouped[sample.image_id], dtype=np.float32).reshape(-1, 5)
        detections[sample.image_id] = (array[:, :4], array[:, 4])
    return detections


def draw(sample, boxes, hits):
    image = cv2.imread(str(sample.path))
    truth, predicted = image.copy(), image.copy()
    for x0, y0, x1, y1 in sample.boxes.astype(int).tolist():
        cv2.rectangle(truth, (x0, y0), (x1, y1), (0, 255, 255), 2)
    for (x0, y0, x1, y1), hit in zip(boxes.astype(int).tolist(), hits.tolist()):
        cv2.rectangle(predicted, (x0, y0), (x1, y1), (0, 200, 0) if hit else (0, 0, 255), 2)
    return np.hstack([truth, predicted])


def main():
    args = parse_args()
    run = Path(args.run)
    folder = run / f"eval_{args.split}"
    samples = load_split(args.data, args.split)
    detections = load_detections(folder / "predictions.json", samples)
    with open(folder / "metrics.json", encoding="utf-8") as f:
        threshold = json.load(f)["best_f1_threshold"]

    shown = {}
    for sample in samples:
        boxes, scores = detections[sample.image_id]
        keep = scores >= threshold
        boxes, scores = boxes[keep], scores[keep]
        single = {sample.image_id: (boxes, scores)}
        ordered_scores, tp, _, _ = match([sample], single)
        hits = int(tp.sum())
        denominator = len(boxes) + len(sample.boxes)
        f1 = 2 * hits / denominator if denominator else 1.0
        order = np.argsort(-scores, kind="stable")
        shown[sample.image_id] = (f1, denominator, sample, boxes[order], tp)

    ranked = sorted((row for row in shown.values() if row[1] > 0), key=lambda row: row[0])
    out = folder / "vis"
    out.mkdir(exist_ok=True)
    for group, rows in (("worst", ranked[:args.count]), ("best", ranked[::-1][:args.count])):
        for k, (f1, _, sample, boxes, tp) in enumerate(rows):
            cv2.imwrite(str(out / f"{group}_{k}_{sample.file_name}"), draw(sample, boxes, tp))
    print(f"saved {args.count * 2} images to {out}")


if __name__ == "__main__":
    main()
