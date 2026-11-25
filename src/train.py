"""Train the sliding-window detector: linear templates with hard-negative mining, then box regression.

Training images only are used for the templates and the box regressor. The validation split is used for
what the specification calls fine-tuning: score calibration and the NMS threshold. The test split is never
read here.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from common import box_iou, flip_boxes, load_split, nms, read_rgb, save_json
from detector import Component, Detector, best_windows, box_deltas
from features import CELL
from metrics import evaluate

OBJECT_CELLS = 8  # a template sees the object at about 32 x 32 px
NEG_IOU = 0.3
POS_IOU = 0.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="dataset root holding images/ and labels/")
    p.add_argument("--name", required=True, help="experiment name, results go to <out>/<name>")
    p.add_argument("--out", default="runs")
    p.add_argument("--features", default="hog+lbp+color", help="any of hog, lbp, color joined by +")
    p.add_argument("--aspects", type=int, default=3, help="number of aspect-ratio templates")
    p.add_argument("--C", type=float, default=1e-4, help="SVM C (larger = less regularised)")
    p.add_argument("--rounds", type=int, default=3, help="hard-negative mining rounds")
    p.add_argument("--pos-per-gt", type=int, default=2, help="positive windows per box per mirror image")
    p.add_argument("--neg-per-image", type=int, default=12, help="random negatives per image per template")
    p.add_argument("--hard-per-image", type=int, default=12, help="mined negatives per image per template")
    p.add_argument("--max-neg", type=int, default=80000, help="negative pool size per template")
    p.add_argument("--reg-alpha", type=float, default=3000.0, help="box regression ridge penalty, 0 disables it")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="use only the first n images, for quick tests")
    p.add_argument("--keep-pool", action="store_true", help="save the window pool for train_reranker.py")
    return p.parse_args()


def set_thread_limits():
    cv2.setNumThreads(1)


def aspect_components(samples, count):
    """Cluster log aspect ratios; every template keeps about the same area."""
    boxes = np.concatenate([s.boxes for s in samples])
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    log_ratio = np.log((boxes[keep, 2] - boxes[keep, 0]) / (boxes[keep, 3] - boxes[keep, 1]))
    centres = KMeans(count, n_init=10, random_state=0).fit(log_ratio[:, None]).cluster_centers_.ravel()
    components = []
    for centre in sorted(centres.tolist()):
        ratio = np.exp(centre)
        components.append(Component(max(3, round(OBJECT_CELLS * np.sqrt(ratio))),
                                    max(3, round(OBJECT_CELLS / np.sqrt(ratio)))))
    return components


def channel_std(detector, sample, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for level in detector.pyramid(read_rgb(sample.path))[2:]:
        flat = level.feats.reshape(-1, level.feats.shape[2])
        rows.append(flat[rng.integers(len(flat), size=min(300, len(flat)))])
    return np.concatenate(rows)


def random_negatives(detector, levels, gt, per_comp, rng):
    found = [[] for _ in detector.components]
    for c, comp in enumerate(detector.components):
        usable = [l for l, level in enumerate(levels)
                  if level.feats.shape[0] >= comp.win_h and level.feats.shape[1] >= comp.win_w]
        for _ in range(per_comp * 3):
            if len(found[c]) >= per_comp:
                break
            level = levels[usable[rng.integers(len(usable))]]
            row = int(rng.integers(level.feats.shape[0] - comp.win_h + 1))
            col = int(rng.integers(level.feats.shape[1] - comp.win_w + 1))
            box = detector.window_box(comp, level, np.array([row]), np.array([col]))
            if len(gt) == 0 or box_iou(box, gt).max() < NEG_IOU:
                found[c].append(level.feats[row:row + comp.win_h, col:col + comp.win_w].ravel())
    return found


def first_pass(detector, sample, pos_per_gt, neg_per_image, seed):
    """Positive windows (with their regression targets) from the image and its mirror, plus random negatives."""
    set_thread_limits()
    rng = np.random.default_rng(seed)
    rgb = read_rgb(sample.path)
    positives = [[] for _ in detector.components]
    negatives = None
    for mirrored in (False, True):
        image = rgb[:, ::-1].copy() if mirrored else rgb
        gts = flip_boxes(sample.boxes, rgb.shape[1]) if mirrored else sample.boxes
        levels = detector.pyramid(image)
        for gt in gts:
            for _, c, l, row, col in best_windows(detector, levels, gt, pos_per_gt, POS_IOU):
                comp = detector.components[c]
                level = levels[l]
                window = detector.window_box(comp, level, np.array([row]), np.array([col]))
                positives[c].append((level.feats[row:row + comp.win_h, col:col + comp.win_w].ravel(),
                                     box_deltas(window, gt[None])[0]))
        if not mirrored:
            negatives = random_negatives(detector, levels, gts, neg_per_image, rng)
    return positives, negatives


def mine_hard(detector, sample, per_comp, margin):
    """False positives of the current templates, best first, limited per template."""
    set_thread_limits()
    levels = detector.pyramid(read_rgb(sample.path))
    cands = detector.proposals(levels)
    if len(sample.boxes):
        cands = cands.take(box_iou(cands.boxes, sample.boxes).max(axis=1) < NEG_IOU)
    cands = cands.take(cands.scores > margin)
    mined = []
    for c in range(len(detector.components)):
        chosen = np.flatnonzero(cands.comp == c)[:per_comp]
        part = cands.take(chosen)
        features = detector.component_features(part, levels, c)
        mined.append((features, part.scores))
    return mined


def fit_svm(X, y, C, start=None):
    """L2-regularised squared-hinge SVM solved with L-BFGS in the primal.

    LinearSVC would copy the dense data into liblinear's sparse format, about 16 bytes per value (16 GB for
    200k windows of 5000 dimensions). Working directly on the float32 matrix avoids that copy, and starting
    from the previous round's weights makes each mining round cheap.
    """
    n_features = X.shape[1]
    labels = y.astype(np.float32)

    def objective(theta):
        w = theta[:-1].astype(np.float32)
        b = float(theta[-1])
        slack = 1 - labels * (X @ w + b)
        active = slack > 0
        loss = 0.5 * float(w @ w) + C * float((slack[active] ** 2).sum())
        grad_w = w.astype(np.float64) - 2 * C * (X[active].T @ (labels[active] * slack[active]))
        grad_b = -2 * C * float((labels[active] * slack[active]).sum())
        return loss, np.append(grad_w, grad_b)

    theta0 = np.zeros(n_features + 1) if start is None else start
    with np.errstate(all="ignore"):  # Apple's Accelerate raises false floating-point warnings in matmul
        result = minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 300, "maxcor": 20})
    return result.x


def set_template(comp, theta, depth):
    comp.weights = theta[:-1].reshape(comp.win_h, comp.win_w, depth).astype(np.float32)
    comp.bias = float(theta[-1])


def fit_regressor(X, deltas, alpha):
    """Ridge regression from window features to box deltas, solved in closed form."""
    mean_x = X.mean(axis=0)
    mean_y = deltas.mean(axis=0)
    centred = (X - mean_x).astype(np.float64)
    gram = centred.T @ centred
    gram[np.diag_indices_from(gram)] += alpha
    weights = np.linalg.solve(gram, centred.T @ (deltas - mean_y))
    return np.vstack([weights, mean_y - mean_x @ weights]).astype(np.float32)


def raw_predict(detector, sample):
    set_thread_limits()
    return sample.image_id, detector.predict_raw(read_rgb(sample.path))


def calibration_rows(detector, sample):
    """score_terms and a matched-a-bird label for every candidate of one validation image."""
    set_thread_limits()
    levels = detector.pyramid(read_rgb(sample.path))
    cands = detector.proposals(levels)
    rows = {}
    boxes, _ = detector.refine(cands, levels)
    best = box_iou(boxes, sample.boxes).max(axis=1) if len(sample.boxes) else np.zeros(len(boxes))
    for c, comp in enumerate(detector.components):
        mask = cands.comp == c
        if mask.any():
            features = detector.component_features(cands, levels, c)
            rows[c] = (detector.score_terms(comp, cands.scores[mask], boxes[mask], features)[:, :-1],
                       best[mask] >= POS_IOU)
    return rows


def calibrate(detector, val, workers):
    """Fit the final logistic score of each template on validation windows."""
    for comp in detector.components:
        comp.calibration = None
    parts = Parallel(n_jobs=workers)(delayed(calibration_rows)(detector, s) for s in val)
    for c, comp in enumerate(detector.components):
        terms = np.concatenate([p[c][0] for p in parts if c in p])
        labels = np.concatenate([p[c][1] for p in parts if c in p])
        mean, std = terms.mean(axis=0), terms.std(axis=0) + 1e-6
        model = LogisticRegression(C=1.0, max_iter=500).fit((terms - mean) / std, labels)
        weights = model.coef_[0] / std
        comp.calibration = np.append(weights, model.intercept_[0] - float(weights @ mean))


def tune_nms(detector, val, workers):
    raw = dict(Parallel(n_jobs=workers)(delayed(raw_predict)(detector, s) for s in val))
    table = {}
    for iou in (0.2, 0.3, 0.4, 0.5, 0.6):
        detections = {}
        for image_id, (boxes, scores) in raw.items():
            keep = nms(boxes, scores, iou)[:detector.max_det]
            detections[image_id] = (boxes[keep], scores[keep])
        table[iou] = evaluate(val, detections)["mAP50"]
    detector.nms_iou = max(table, key=table.get)
    return table


def main():
    args = parse_args()
    set_thread_limits()
    start = time.perf_counter()
    out = Path(args.out) / args.name
    train = load_split(args.data, "train")
    val = load_split(args.data, "val")
    if args.limit:
        train, val = train[:args.limit], val[:max(args.limit // 4, 20)]
    names = args.features.split("+")
    log = {"args": vars(args), "rounds": []}

    detector = Detector(names, aspect_components(train, args.aspects))
    print("templates (cells w x h):", [(c.inner_w, c.inner_h) for c in detector.components])

    rng = np.random.default_rng(0)
    sampled = rng.choice(len(train), size=min(80, len(train)), replace=False).tolist()
    rows = Parallel(n_jobs=args.workers)(delayed(channel_std)(detector, train[k], k) for k in sampled)
    detector.channel_scale = np.maximum(np.concatenate(rows).std(axis=0), 1e-3).astype(np.float32)

    parts = Parallel(n_jobs=args.workers, verbose=1)(
        delayed(first_pass)(detector, s, args.pos_per_gt, args.neg_per_image, k) for k, s in enumerate(train))
    pools = []
    for c in range(len(detector.components)):
        rows = [row for positives, _ in parts for row in positives[c]]
        pools.append({"pos": np.stack([r[0] for r in rows]),
                      "deltas": np.stack([r[1] for r in rows]),
                      "neg": np.stack([x for _, negatives in parts for x in negatives[c]])})
    del parts
    print("positives per template:", [len(p["pos"]) for p in pools], flush=True)

    thetas = [None] * len(pools)
    for round_index in range(args.rounds + 1):
        round_start = time.perf_counter()
        for c, (comp, pool) in enumerate(zip(detector.components, pools)):
            X = np.concatenate([pool["pos"], pool["neg"]])
            y = np.r_[np.ones(len(pool["pos"])), -np.ones(len(pool["neg"]))]
            thetas[c] = fit_svm(X, y, args.C, thetas[c])
            set_template(comp, thetas[c], detector.depth)
            del X
        fit_seconds = time.perf_counter() - round_start

        subset = val[:100]
        raw = dict(Parallel(n_jobs=args.workers)(delayed(raw_predict)(detector, s) for s in subset))
        detections = {}
        for image_id, (boxes, scores) in raw.items():
            keep = nms(boxes, scores, detector.nms_iou)[:detector.max_det]
            detections[image_id] = (boxes[keep], scores[keep])
        entry = {"round": round_index, "fit_seconds": round(fit_seconds, 1),
                 "negatives": [len(p["neg"]) for p in pools], "val100_mAP50": evaluate(subset, detections)["mAP50"]}

        if round_index < args.rounds:
            mined = Parallel(n_jobs=args.workers)(
                delayed(mine_hard)(detector, s, args.hard_per_image, -1.0) for s in train)
            for c, comp in enumerate(detector.components):
                pool = pools[c]
                old_scores = pool["neg"] @ comp.weights.ravel() + comp.bias
                new = np.concatenate([m[c][0] for m in mined])
                new_scores = np.concatenate([m[c][1] for m in mined])
                negatives = np.concatenate([pool["neg"], new])
                scores = np.r_[old_scores, new_scores]
                keep = np.argsort(-scores)[:args.max_neg]
                pool["neg"] = negatives[keep]
                entry.setdefault("mined", []).append(len(new))
        entry["round_seconds"] = round(time.perf_counter() - round_start, 1)
        log["rounds"].append(entry)
        print(entry, flush=True)

    if args.reg_alpha > 0:
        for comp, pool in zip(detector.components, pools):
            comp.regressor = fit_regressor(pool["pos"], pool["deltas"], args.reg_alpha)
    log["train_seconds"] = time.perf_counter() - start

    # Save before tuning: a failed calibration must not cost the whole training run.
    detector.save(out / "model.pkl")
    save_json(out / "train_log.json", log)
    if args.keep_pool:
        arrays = {}
        for c, pool in enumerate(pools):
            arrays.update({f"{key}{c}": value for key, value in pool.items()})
        np.savez(out / "pool.npz", **arrays)

    tune_start = time.perf_counter()
    calibrate(detector, val, args.workers)
    log["nms_table"] = {str(k): v for k, v in tune_nms(detector, val, args.workers).items()}
    log["tune_seconds"] = time.perf_counter() - tune_start
    log["nms_iou"] = detector.nms_iou
    print("val mAP50 by NMS IoU:", log["nms_table"], "->", detector.nms_iou, flush=True)

    detector.save(out / "model.pkl")
    save_json(out / "train_log.json", log)
    print(f"saved {out} in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
