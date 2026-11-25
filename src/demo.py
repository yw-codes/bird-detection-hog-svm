"""Live demo: detect birds in one or more images and show the result.

    python demo.py --run runs/c1e3_gbdt --image photo.jpg [photo2.jpg ...]

For each image this prints how many birds were found and how long it took, saves a picture of the result
and shows it in a window (press any key for the next image). If a YOLO label file with the same name sits
next to the image, or in the matching labels/ folder of the dataset, the labelled birds are drawn too:
yellow for labelled birds, green for a detection that matches one (IoU >= 0.5), red for one that does not.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from common import box_iou, read_rgb
from detector import Detector

PANEL = 640  # display size of each half of the picture, in pixels


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="folder holding model.pkl (and eval_val/metrics.json)")
    p.add_argument("--image", required=True, nargs="+", help="one or more image files")
    p.add_argument("--threshold", type=float, help="score threshold; default: the best-F1 threshold on val")
    p.add_argument("--out", default="demo_out", help="folder for the saved pictures")
    p.add_argument("--no-show", action="store_true", help="only save the pictures, open no window")
    return p.parse_args()


def default_threshold(run):
    path = Path(run) / "eval_val" / "metrics.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return float(json.load(f)["best_f1_threshold"])
    return 0.5


def find_label(image_path):
    candidates = [image_path.with_suffix(".txt")]
    parts = list(image_path.parts)
    if "images" in parts:
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        candidates.append(Path(*parts).with_suffix(".txt"))
    return next((c for c in candidates if c.exists()), None)


def read_label(path, width, height):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                cx, cy, w, h = (float(v) for v in parts[1:5])
                rows.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
    return np.array(rows, dtype=np.float32).reshape(-1, 4)


def match(boxes, truth, iou_thr=0.5):
    """Greedy matching in score order; returns a hit flag per detection."""
    hits = np.zeros(len(boxes), dtype=bool)
    if len(boxes) == 0 or len(truth) == 0:
        return hits
    iou = box_iou(boxes, truth)
    taken = np.zeros(len(truth), dtype=bool)
    for i in range(len(boxes)):
        candidates = np.where(taken, -1.0, iou[i])
        j = int(candidates.argmax())
        if candidates[j] >= iou_thr:
            hits[i], taken[j] = True, True
    return hits


def panel(image, title):
    scale = PANEL / max(image.shape[:2])
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    header = np.full((40, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([header, image])


def draw(bgr, boxes, colours):
    out = bgr.copy()
    for (x0, y0, x1, y1), colour in zip(boxes.astype(int).tolist(), colours):
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, 2)
    return out


def main():
    args = parse_args()
    detector = Detector.load(Path(args.run) / "model.pkl")
    threshold = default_threshold(args.run) if args.threshold is None else args.threshold
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"model: {args.run}  (score threshold {threshold:.2f})")

    for name in args.image:
        path = Path(name)
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"cannot read {path}, skipped")
            continue
        start = time.perf_counter()
        boxes, scores = detector.predict(read_rgb(path))
        seconds = time.perf_counter() - start
        keep = scores >= threshold
        boxes, scores = boxes[keep], scores[keep]
        print(f"\n{path.name}: found {len(boxes)} birds in {seconds:.2f} s")

        label = find_label(path)
        if label is None:
            picture = panel(draw(bgr, boxes, [(0, 200, 0)] * len(boxes)), f"Detected: {len(boxes)} birds")
        else:
            truth = read_label(label, bgr.shape[1], bgr.shape[0])
            hits = match(boxes, truth)
            correct = int(hits.sum())
            print(f"  labelled birds: {len(truth)}, correct: {correct}, "
                  f"false detections: {len(boxes) - correct}, missed: {len(truth) - correct}")
            left = panel(draw(bgr, truth, [(0, 255, 255)] * len(truth)), f"Ground truth: {len(truth)} birds")
            right = panel(draw(bgr, boxes, [(0, 200, 0) if h else (0, 0, 255) for h in hits]),
                          f"Detected: {len(boxes)} ({correct} correct)")
            picture = np.hstack([left, np.full((left.shape[0], 8, 3), 255, dtype=np.uint8), right])

        saved = out / f"demo_{path.stem}.jpg"
        cv2.imwrite(str(saved), picture)
        print(f"  saved {saved}")
        if not args.no_show:
            try:
                cv2.imshow("Traditional detector (HOG + LBP + colour, SVM + GBDT)", picture)
                cv2.waitKey(0)
            except cv2.error:
                print("  no display available; open the saved picture instead")
    if not args.no_show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
