"""Second stage: a non-linear classifier over the windows the linear templates cannot separate.

Trees are too slow to scan every window of every pyramid level, so the template stage stays as a fast
filter (a two-stage cascade, as in Viola & Jones, 2001) and only its proposals reach the trees. The training
windows are the pool left by hard-negative mining, which matches what the second stage sees at run time.
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from common import load_split, save_json
from detector import Detector
from train import calibrate, tune_nms


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="template run folder (trained with --keep-pool)")
    p.add_argument("--data", required=True)
    p.add_argument("--kind", default="rf", choices=["rf", "gbdt"])
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def make_model(kind, trees):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=trees, min_samples_leaf=2, max_features="sqrt", n_jobs=-1,
                                      class_weight="balanced_subsample", random_state=0)
    # Looking at 10 % of the features per split keeps thousands of channels affordable.
    return HistGradientBoostingClassifier(max_iter=trees, learning_rate=0.1, max_leaf_nodes=31, max_features=0.1,
                                          class_weight="balanced", early_stopping=False, random_state=0)


def main():
    args = parse_args()
    run = Path(args.run)
    out = run.parent / f"{run.name}_{args.kind}"
    detector = Detector.load(run / "model.pkl")
    start = time.perf_counter()
    with np.load(run / "pool.npz") as pool:
        for c, comp in enumerate(detector.components):
            positives, negatives = pool[f"pos{c}"], pool[f"neg{c}"]
            X = np.concatenate([positives, negatives])
            y = np.r_[np.ones(len(positives), dtype=np.int8), np.zeros(len(negatives), dtype=np.int8)]
            comp.reranker = make_model(args.kind, args.trees).fit(X, y)
            if args.kind == "rf":
                comp.reranker.n_jobs = 1  # evaluation already runs one process per image
            print(f"template {c}: {len(positives)} + {len(negatives)} windows, "
                  f"{time.perf_counter() - start:.0f} s", flush=True)
    fit_seconds = time.perf_counter() - start

    val = load_split(args.data, "val")
    calibrate(detector, val, args.workers)
    table = tune_nms(detector, val, args.workers)
    detector.save(out / "model.pkl")

    log = {"args": vars(args), "reranker_seconds": fit_seconds, "nms_iou": detector.nms_iou,
           "nms_table": {str(k): v for k, v in table.items()}}
    template_log = run / "train_log.json"
    if template_log.exists():
        shutil.copy(template_log, out / "template_log.json")
    save_json(out / "train_log.json", log)
    print(f"saved {out}; val mAP50 by NMS IoU {table}")


if __name__ == "__main__":
    main()
