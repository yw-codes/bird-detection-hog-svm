"""mAP@0.5 and the per-density / per-size breakdowns, without any detection framework."""

import numpy as np

from common import DENSITY_BINS, box_iou

SIZE_BINS = (("small (<32 px)", 0, 32), ("medium (32-96 px)", 32, 96), ("large (>96 px)", 96, 1e9))


def match(samples, detections, iou_thr=0.5):
    """COCO-style greedy matching: best-scoring detections claim the best free ground-truth box.

    Returns scores, true-positive flags and the owning sample index for every detection, plus one list per
    sample with the best score at which each ground-truth box was found (NaN if it never was).
    """
    all_scores, all_tp, all_index, found = [], [], [], []
    for s, sample in enumerate(samples):
        boxes, scores = detections[sample.image_id]
        order = np.argsort(-scores, kind="stable")
        boxes, scores = boxes[order], scores[order]
        iou = box_iou(boxes, sample.boxes)
        free = np.ones(len(sample.boxes), dtype=bool)
        tp = np.zeros(len(boxes), dtype=bool)
        found_at = np.full(len(sample.boxes), np.nan)
        for d in range(len(boxes)):
            if not free.any():
                break
            overlap = np.where(free, iou[d], -1)
            g = int(overlap.argmax())
            if overlap[g] >= iou_thr:
                free[g] = False
                tp[d] = True
                found_at[g] = scores[d]
        all_scores.append(scores)
        all_tp.append(tp)
        all_index.append(np.full(len(boxes), s))
        found.append(found_at)
    return np.concatenate(all_scores), np.concatenate(all_tp), np.concatenate(all_index), found


def pr_curve(scores, tp, positives):
    order = np.argsort(-scores, kind="stable")
    hits = np.cumsum(tp[order])
    misses = np.cumsum(~tp[order])
    return hits / max(positives, 1), hits / np.maximum(hits + misses, 1), scores[order]


def average_precision(recall, precision):
    """101-point interpolated AP, the COCO definition."""
    if len(recall) == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    index = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.where(index < len(envelope), envelope[index.clip(max=len(envelope) - 1)], 0).mean())


def evaluate(samples, detections, iou_thr=0.5):
    """Overall AP, best-F1 operating point, counting error and breakdowns."""
    positives = sum(len(s.boxes) for s in samples)
    scores, tp, owner, found = match(samples, detections, iou_thr)
    recall, precision, ordered = pr_curve(scores, tp, positives)
    result = {"images": len(samples), "gt_boxes": positives, "detections": len(scores),
              "mAP50": average_precision(recall, precision)}

    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best = int(f1.argmax()) if len(f1) else 0
    threshold = float(ordered[best]) if len(f1) else 0.0
    result.update(best_f1=float(f1[best]) if len(f1) else 0.0, best_f1_threshold=threshold,
                  precision=float(precision[best]) if len(f1) else 0.0,
                  recall=float(recall[best]) if len(f1) else 0.0,
                  max_recall=float(recall[-1]) if len(recall) else 0.0)

    counts = np.array([int((detections[s.image_id][1] >= threshold).sum()) for s in samples])
    truth = np.array([len(s.boxes) for s in samples])
    result["count_mae"] = float(np.abs(counts - truth).mean())

    density = {}
    for name in DENSITY_BINS:
        members = [k for k, s in enumerate(samples) if s.density == name]
        chosen = np.isin(owner, members)
        total = int(truth[members].sum()) if members else 0
        row = {"images": len(members), "gt_boxes": total, "count_mae": float(np.abs(counts - truth)[members].mean())
               if members else None}
        if total:
            row["AP50"] = average_precision(*pr_curve(scores[chosen], tp[chosen], total)[:2])
        else:
            row["false_positives_per_image"] = float((scores[chosen] >= threshold).sum() / max(len(members), 1))
        density[name] = row
    result["by_density"] = density

    sides = np.concatenate([np.sqrt((s.boxes[:, 2] - s.boxes[:, 0]) * (s.boxes[:, 3] - s.boxes[:, 1]))
                            for s in samples]) if positives else np.zeros(0)
    found_at = np.concatenate(found) if positives else np.zeros(0)
    by_size = {}
    for name, low, high in SIZE_BINS:
        mask = (sides >= low) & (sides < high)
        by_size[name] = {"gt_boxes": int(mask.sum()),
                         "recall_at_best_f1": float((found_at[mask] >= threshold).mean()) if mask.any() else None,
                         "max_recall": float((~np.isnan(found_at[mask])).mean()) if mask.any() else None}
    result["by_size"] = by_size
    return result
