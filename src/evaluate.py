"""Run a trained detector on a split and write metrics.json, predictions.json and pr_curve.png.

The test split is locked behind --final: use it once, after every choice has been made on validation.
"""

import argparse
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from common import load_split, predictions_to_coco, read_rgb, save_json
from detector import Detector
from metrics import evaluate, match, pr_curve


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run folder holding model.pkl")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--final", action="store_true", help="required for --split test")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timing-images", type=int, default=30, help="images timed one by one on a single thread")
    return p.parse_args()


def detect(detector, sample):
    cv2.setNumThreads(1)
    return sample.image_id, detector.predict(read_rgb(sample.path))


def seconds_per_image(detector, samples):
    """Single-thread latency, including JPEG decoding and the final NMS."""
    cv2.setNumThreads(1)
    start = time.perf_counter()
    for sample in samples:
        detector.predict(read_rgb(sample.path))
    return (time.perf_counter() - start) / len(samples)


def main():
    args = parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("The test split is for the final evaluation only. Pass --final if this is that run.")
    run = Path(args.run)
    detector = Detector.load(run / "model.pkl")
    samples = load_split(args.data, args.split)

    detections = dict(Parallel(n_jobs=args.workers, verbose=1)(delayed(detect)(detector, s) for s in samples))
    metrics = evaluate(samples, detections)
    metrics["seconds_per_image"] = seconds_per_image(detector, samples[:args.timing_images])
    metrics["split"] = args.split

    out = run / f"eval_{args.split}"
    save_json(out / "metrics.json", metrics)
    save_json(out / "predictions.json", predictions_to_coco(samples, detections))

    scores, tp, _, _ = match(samples, detections)
    recall, precision, _ = pr_curve(scores, tp, metrics["gt_boxes"])
    plt.figure(figsize=(4, 3))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{run.name}: AP50 = {metrics['mAP50']:.3f}")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(out / "pr_curve.png", dpi=150)

    print(f"{args.split}: mAP@0.5 {metrics['mAP50']:.4f}  best F1 {metrics['best_f1']:.3f} "
          f"(P {metrics['precision']:.3f} / R {metrics['recall']:.3f})  "
          f"max recall {metrics['max_recall']:.3f}  {metrics['seconds_per_image']:.3f} s/image")


if __name__ == "__main__":
    main()
