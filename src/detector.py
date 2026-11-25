"""Multi-scale, multi-aspect sliding-window detector built on linear templates.

A Detector holds one Component per aspect ratio. Each component scores every window of every pyramid
level with a linear template. The best windows then go through optional box regression and an optional
non-linear reranker before the final NMS. Detector.predict(rgb) is the only method other code needs.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from common import box_iou, nms
from features import CELL, dense_features, feature_depth

PAD = 4  # pixels of replicated border, so a window can sit on a bird cut off by the image edge
MARGIN = 1  # cells of context kept around the object inside a window
MAX_SCALE = 2.0  # object side of about 16 px
MIN_SCALE = 0.07  # object side of about 450 px
SCALE_STEP = 2 ** (-1 / 3)
PROPOSAL_NMS = 0.8  # only merges near-identical windows; the real NMS comes after the final scores


def default_scales():
    count = int(np.floor(np.log(MIN_SCALE / MAX_SCALE) / np.log(SCALE_STEP))) + 1
    return [MAX_SCALE * SCALE_STEP ** k for k in range(count)]


@dataclass
class Component:
    """One aspect ratio. Sizes are in cells; a window is the object plus MARGIN cells on every side."""
    inner_w: int
    inner_h: int
    weights: np.ndarray = None  # win_h x win_w x D
    bias: float = 0.0
    regressor: np.ndarray = None  # (features + 1) x 4
    calibration: np.ndarray = None  # logistic weights over Detector.score_terms
    reranker: object = None

    @property
    def win_w(self):
        return self.inner_w + 2 * MARGIN

    @property
    def win_h(self):
        return self.inner_h + 2 * MARGIN


@dataclass
class Level:
    feats: np.ndarray
    sx: float
    sy: float


@dataclass
class Candidates:
    """Windows that survived the template stage. All arrays have one row per window."""
    boxes: np.ndarray
    scores: np.ndarray
    comp: np.ndarray
    level: np.ndarray
    row: np.ndarray
    col: np.ndarray

    def __len__(self):
        return len(self.scores)

    def take(self, index):
        return Candidates(self.boxes[index], self.scores[index], self.comp[index],
                          self.level[index], self.row[index], self.col[index])


@dataclass
class Detector:
    feature_names: list
    components: list
    channel_scale: np.ndarray = None
    scales: list = field(default_factory=default_scales)
    nms_iou: float = 0.4
    max_det: int = 300
    pre_nms_top: int = 1500

    @property
    def depth(self):
        return feature_depth(self.feature_names)

    def pyramid(self, rgb):
        height, width = rgb.shape[:2]
        min_side = min(min(c.win_w, c.win_h) for c in self.components) * CELL
        levels = []
        for scale in self.scales:
            new_w, new_h = round(width * scale), round(height * scale)
            if min(new_w, new_h) + 2 * PAD < min_side:
                continue
            method = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            image = cv2.resize(rgb, (new_w, new_h), interpolation=method)
            image = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
            feats = dense_features(image, self.feature_names)
            if self.channel_scale is not None:
                feats /= self.channel_scale
            levels.append(Level(feats, new_w / width, new_h / height))
        return levels

    def window_box(self, comp, level, row, col):
        """Object box in original image pixels for window(s) at cell (row, col)."""
        x0 = (col + MARGIN) * CELL - PAD
        y0 = (row + MARGIN) * CELL - PAD
        return np.stack([x0 / level.sx, y0 / level.sy,
                         (x0 + comp.inner_w * CELL) / level.sx,
                         (y0 + comp.inner_h * CELL) / level.sy], axis=-1).astype(np.float32)

    def template_scores(self, comp, level):
        """Correlate the template with one level; row r, col c is the window starting at that cell."""
        rows, cols, depth = level.feats.shape
        out_rows, out_cols = rows - comp.win_h + 1, cols - comp.win_w + 1
        if out_rows <= 0 or out_cols <= 0:
            return None
        total = np.full((out_rows, out_cols), comp.bias, dtype=np.float32)
        for d in range(depth):
            response = cv2.filter2D(level.feats[:, :, d], cv2.CV_32F, comp.weights[:, :, d],
                                    anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
            total += response[:out_rows, :out_cols]
        return total

    def candidates(self, levels, top):
        """Best `top` windows over all levels and components, by raw template score."""
        per_level = max(top, 50)
        parts = []
        for c, comp in enumerate(self.components):
            for l, level in enumerate(levels):
                scores = self.template_scores(comp, level)
                if scores is None:
                    continue
                flat = scores.ravel()
                index = np.arange(flat.size) if flat.size <= per_level \
                    else np.argpartition(-flat, per_level)[:per_level]
                row, col = np.unravel_index(index, scores.shape)
                parts.append((flat[index], np.full(len(index), c), np.full(len(index), l), row, col))
        scores, comp, lev, row, col = (np.concatenate(p) for p in zip(*parts))
        best = np.argsort(-scores)[:top]
        scores, comp, lev, row, col = scores[best], comp[best], lev[best], row[best], col[best]
        boxes = np.zeros((len(best), 4), dtype=np.float32)
        for c, l in set(zip(comp.tolist(), lev.tolist())):
            mask = (comp == c) & (lev == l)
            boxes[mask] = self.window_box(self.components[c], levels[l], row[mask], col[mask])
        return Candidates(boxes, scores, comp, lev, row, col)

    def proposals(self, levels):
        """Template-stage output: the best windows, with near-duplicates removed."""
        cands = self.candidates(levels, self.pre_nms_top)
        return cands.take(nms(cands.boxes, cands.scores, PROPOSAL_NMS))

    def component_features(self, cands, levels, c):
        """Flattened window features of the candidates that belong to component `c`, in candidate order."""
        comp = self.components[c]
        positions = np.flatnonzero(cands.comp == c)
        features = np.zeros((len(positions), comp.win_h * comp.win_w * self.depth), dtype=np.float32)
        for n, p in enumerate(positions.tolist()):
            r, k = int(cands.row[p]), int(cands.col[p])
            features[n] = levels[cands.level[p]].feats[r:r + comp.win_h, k:k + comp.win_w].ravel()
        return features

    def score_terms(self, comp, cands_scores, boxes, features):
        """Inputs of the final logistic score: template score, log box side, reranker logit, 1."""
        terms = [cands_scores, np.log(np.sqrt(box_area(boxes)) + 1)]
        if comp.reranker is not None:
            prob = comp.reranker.predict_proba(features)[:, 1].clip(1e-4, 1 - 1e-4)
            terms.append(np.log(prob / (1 - prob)))
        terms.append(np.ones(len(boxes)))
        return np.stack(terms, axis=1)

    def refine(self, cands, levels):
        """Box regression (when trained), then the calibrated final score of every candidate."""
        boxes = cands.boxes.copy()
        scores = cands.scores.astype(np.float64)
        for c, comp in enumerate(self.components):
            mask = cands.comp == c
            if not mask.any() or (comp.regressor is None and comp.calibration is None):
                continue
            features = self.component_features(cands, levels, c)
            if comp.regressor is not None:
                boxes[mask] = apply_regression(boxes[mask], features, comp.regressor)
            if comp.calibration is not None:
                logit = (self.score_terms(comp, cands.scores[mask], boxes[mask], features)
                         * comp.calibration).sum(axis=1)
                scores[mask] = 1 / (1 + np.exp(-logit))
        return boxes, scores.astype(np.float32)

    def predict_raw(self, rgb):
        """Refined boxes and scores before the final NMS."""
        height, width = rgb.shape[:2]
        levels = self.pyramid(rgb)
        boxes, scores = self.refine(self.proposals(levels), levels)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        return boxes, scores

    def predict(self, rgb, nms_iou=None):
        """RGB float image -> (N x 4 xyxy boxes in pixels, N scores), best first."""
        boxes, scores = self.predict_raw(rgb)
        keep = nms(boxes, scores, self.nms_iou if nms_iou is None else nms_iou)[:self.max_det]
        return boxes[keep], scores[keep]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]).clip(1) * (boxes[:, 3] - boxes[:, 1]).clip(1)


def box_deltas(windows, targets):
    """R-CNN box regression targets (Girshick et al., 2014)."""
    w, h = windows[:, 2] - windows[:, 0], windows[:, 3] - windows[:, 1]
    cx, cy = windows[:, 0] + w / 2, windows[:, 1] + h / 2
    tw, th = (targets[:, 2] - targets[:, 0]).clip(1), (targets[:, 3] - targets[:, 1]).clip(1)
    tx, ty = targets[:, 0] + tw / 2, targets[:, 1] + th / 2
    return np.stack([(tx - cx) / w, (ty - cy) / h, np.log(tw / w), np.log(th / h)], axis=1)


def apply_regression(boxes, features, regressor):
    with np.errstate(all="ignore"):  # Apple's Accelerate raises false floating-point warnings in matmul
        deltas = features @ regressor[:-1] + regressor[-1]
    deltas[:, 2:] = deltas[:, 2:].clip(-1, 1)
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    cx = boxes[:, 0] + w / 2 + deltas[:, 0] * w
    cy = boxes[:, 1] + h / 2 + deltas[:, 1] * h
    new_w, new_h = w * np.exp(deltas[:, 2]), h * np.exp(deltas[:, 3])
    return np.stack([cx - new_w / 2, cy - new_h / 2, cx + new_w / 2, cy + new_h / 2], axis=1)


def best_windows(detector, levels, gt, count, min_iou):
    """For one ground-truth box, the `count` best-aligned (component, level, row, col, iou) windows."""
    found = []
    for c, comp in enumerate(detector.components):
        for l, level in enumerate(levels):
            rows, cols = level.feats.shape[:2]
            max_row, max_col = rows - comp.win_h, cols - comp.win_w
            if max_row < 0 or max_col < 0:
                continue
            col = np.clip(round((gt[0] * level.sx + PAD) / CELL - MARGIN), 0, max_col)
            row = np.clip(round((gt[1] * level.sy + PAD) / CELL - MARGIN), 0, max_row)
            box = detector.window_box(comp, level, np.array([row]), np.array([col]))
            iou = float(box_iou(box, gt[None])[0, 0])
            if iou >= min_iou:
                found.append((iou, c, l, int(row), int(col)))
    found.sort(reverse=True)
    return found[:count]
