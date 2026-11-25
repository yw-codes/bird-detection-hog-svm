"""Beach Bird dataset (YOLO labels), box helpers and the shared prediction file format."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "val", "test")
DENSITY_BINS = ("0", "1-5", "6-20", "21-50", "51+")


def density_bin(count):
    """Same density bins as the dataset README (and its stratified split)."""
    if count == 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 20:
        return "6-20"
    if count <= 50:
        return "21-50"
    return "51+"


@dataclass
class Sample:
    image_id: int
    file_name: str
    path: Path
    boxes: np.ndarray  # N x 4, xyxy in pixels
    density: str


def load_split(root, split):
    """Read one split. Negative images have an empty label file and get a 0 x 4 box array."""
    root = Path(root)
    samples = []
    for image_path in sorted((root / "images" / split).glob("*.jpg")):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        rows = []
        with open(root / "labels" / split / f"{image_path.stem}.txt", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, w, h = (float(v) for v in parts[1:5])
                rows.append([(cx - w / 2) * width, (cy - h / 2) * height,
                             (cx + w / 2) * width, (cy + h / 2) * height])
        boxes = np.array(rows, dtype=np.float32).reshape(-1, 4)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        samples.append(Sample(int(image_path.stem.rsplit("_", 1)[1]), image_path.name, image_path,
                              boxes, density_bin(len(boxes))))
    if not samples:
        raise FileNotFoundError(f"no images in {root / 'images' / split}")
    return samples


def read_rgb(path):
    """Image as float32 RGB in [0, 1]."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def flip_boxes(boxes, width):
    flipped = boxes.copy()
    flipped[:, 0] = width - boxes[:, 2]
    flipped[:, 2] = width - boxes[:, 0]
    return flipped


def box_iou(a, b):
    """Pairwise IoU between N x 4 and M x 4 xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x1 - x0).clip(0) * (y1 - y0).clip(0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms(boxes, scores, iou_thr):
    """Greedy non-maximum suppression; returns kept indices, best score first."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        best = order[0]
        keep.append(best)
        if len(order) == 1:
            break
        rest = order[1:]
        order = rest[box_iou(boxes[best:best + 1], boxes[rest])[0] <= iou_thr]
    return np.array(keep, dtype=np.int64)


def predictions_to_coco(samples, detections):
    """COCO results list; `detections` maps image_id to (boxes xyxy, scores)."""
    results = []
    for sample in samples:
        boxes, scores = detections[sample.image_id]
        for (x0, y0, x1, y1), score in zip(boxes.tolist(), scores.tolist()):
            results.append({"image_id": sample.image_id, "file_name": sample.file_name, "category_id": 1,
                            "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                            "score": round(score, 5)})
    return results


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
