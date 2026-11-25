"""Dense handcrafted features computed on a CELL x CELL pixel grid.

Every descriptor returns an H x W x D array with one vector per cell, so a detection window is simply a
crop of that array. This is what makes the sliding-window search cheap: features are computed once per
pyramid level instead of once per window. All three descriptors are implemented here with NumPy; OpenCV
is only used for resizing, colour conversion and blurring.
"""

import cv2
import numpy as np

CELL = 4
EPS = 1e-4
GROUPS = {"hog": 31, "lbp": 10, "color": 6}
SURROUND_SIGMA = 6.0  # in cells, i.e. 24 px at the current pyramid level: wider than a 40 px window


def cell_shape(image):
    return image.shape[0] // CELL, image.shape[1] // CELL


def hog_channels(rgb):
    """Felzenszwalb et al. (2010) HOG: 18 signed + 9 unsigned orientations + 4 texture channels."""
    rows, cols = cell_shape(rgb)
    rgb = rgb[:rows * CELL, :cols * CELL]

    # Centred gradients, keeping the colour channel with the strongest response (Dalal & Triggs, 2005).
    gx = np.zeros_like(rgb)
    gy = np.zeros_like(rgb)
    gx[:, 1:-1] = rgb[:, 2:] - rgb[:, :-2]
    gy[1:-1, :] = rgb[2:, :] - rgb[:-2, :]
    strongest = (gx ** 2 + gy ** 2).argmax(axis=2)[..., None]
    gx = np.take_along_axis(gx, strongest, axis=2)[..., 0]
    gy = np.take_along_axis(gy, strongest, axis=2)[..., 0]
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Each pixel votes for its two nearest orientation bins.
    position = (np.arctan2(gy, gx) % (2 * np.pi)) / (2 * np.pi) * 18
    low = position.astype(np.int32) % 18
    high = (low + 1) % 18
    weight_high = position - np.floor(position)
    cell_index = (np.arange(rows * CELL) // CELL)[:, None] * cols + (np.arange(cols * CELL) // CELL)[None, :]
    cell_index = cell_index * 18
    size = rows * cols * 18
    hist = (np.bincount((cell_index + low).ravel(), (magnitude * (1 - weight_high)).ravel(), size)
            + np.bincount((cell_index + high).ravel(), (magnitude * weight_high).ravel(), size))
    hist = hist.reshape(rows, cols, 18).astype(np.float32)

    # A small spatial blur stands in for trilinear interpolation between neighbouring cells.
    kernel = np.array([0.125, 0.75, 0.125], dtype=np.float32)
    for k in range(18):
        hist[:, :, k] = cv2.sepFilter2D(hist[:, :, k], -1, kernel, kernel, borderType=cv2.BORDER_REPLICATE)

    unsigned = hist[:, :, :9] + hist[:, :, 9:]
    energy = np.pad((unsigned ** 2).sum(axis=2), 1, mode="edge")
    block = energy[:-1, :-1] + energy[:-1, 1:] + energy[1:, :-1] + energy[1:, 1:]
    norm = 1 / np.sqrt(block + EPS)
    norms = (norm[:-1, :-1, None], norm[:-1, 1:, None], norm[1:, :-1, None], norm[1:, 1:, None])

    signed_part = sum(np.minimum(hist * n, 0.2) for n in norms) * 0.5
    clipped = [np.minimum(unsigned * n, 0.2) for n in norms]
    unsigned_part = sum(clipped) * 0.5
    texture = np.stack([c.sum(axis=2) for c in clipped], axis=2) * 0.2357
    return np.concatenate([signed_part, unsigned_part, texture], axis=2)


def _lbp_lookup():
    """Rotation-invariant uniform LBP (Ojala et al., 2002): 256 codes mapped to 10 classes."""
    table = np.zeros(256, dtype=np.int64)
    for code in range(256):
        bits = [(code >> k) & 1 for k in range(8)]
        transitions = sum(bits[k] != bits[(k + 1) % 8] for k in range(8))
        table[code] = sum(bits) if transitions <= 2 else 9
    return table


LBP_TABLE = _lbp_lookup()


def lbp_channels(rgb):
    """Per-cell histogram of LBP(8, 1) classes, as fractions of the cell's pixels."""
    rows, cols = cell_shape(rgb)
    grey = cv2.cvtColor(rgb[:rows * CELL, :cols * CELL], cv2.COLOR_RGB2GRAY)
    padded = np.pad(grey, 1, mode="edge")
    centre = padded[1:-1, 1:-1]
    offsets = ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0))
    height, width = padded.shape
    code = np.zeros(centre.shape, dtype=np.int64)
    for bit, (dy, dx) in enumerate(offsets):
        neighbour = padded[dy:height - 2 + dy, dx:width - 2 + dx]
        code |= (neighbour >= centre).astype(np.int64) << bit
    cell_index = (np.arange(rows * CELL) // CELL)[:, None] * cols + (np.arange(cols * CELL) // CELL)[None, :]
    hist = np.bincount((cell_index * 10 + LBP_TABLE[code]).ravel(), minlength=rows * cols * 10)
    return hist.reshape(rows, cols, 10).astype(np.float32) / (CELL * CELL)


def color_channels(rgb):
    """Centre-surround colour contrast in CIE Luv.

    Backgrounds range from orange sunsets to blue sea, so absolute colour says little about whether a
    region is a bird. The difference from the blurred neighbourhood does: `d` lets a template learn "darker
    than the surroundings" and `|d|` lets it learn "different from the surroundings" for birds that can be
    either lighter or darker, which a linear model could not express from `d` alone.
    """
    rows, cols = cell_shape(rgb)
    luv = cv2.cvtColor(rgb[:rows * CELL, :cols * CELL], cv2.COLOR_RGB2Luv) / 100.0
    cells = cv2.resize(luv, (cols, rows), interpolation=cv2.INTER_AREA)
    surround = cv2.GaussianBlur(cells, (0, 0), SURROUND_SIGMA, borderType=cv2.BORDER_REPLICATE)
    difference = cells - surround
    return np.concatenate([difference, np.abs(difference)], axis=2)


EXTRACTORS = {"hog": hog_channels, "lbp": lbp_channels, "color": color_channels}


def dense_features(rgb, names):
    """Stack the requested descriptors into one H x W x D float32 array."""
    return np.concatenate([EXTRACTORS[name](rgb) for name in names], axis=2).astype(np.float32)


def feature_depth(names):
    return sum(GROUPS[name] for name in names)
