"""Streaming region-mask metrics. No dataset-sized probability array is kept."""

import numpy as np


def divide(numerator, denominator, empty=0.0):
    a, b = np.broadcast_arrays(np.asarray(numerator, dtype=np.float64), np.asarray(denominator, dtype=np.float64))
    return np.divide(a, b, out=np.full_like(a, empty), where=b != 0)


def from_counts(tp, fp, fn, tn):
    precision = divide(tp, tp + fp, empty=1.0 if tp + fn == 0 else 0.0)
    recall = divide(tp, tp + fn, empty=1.0)
    f1 = divide(2 * tp, 2 * tp + fp + fn, empty=1.0)
    foreground = divide(tp, tp + fp + fn, empty=1.0)
    background = divide(tn, tn + fp + fn, empty=1.0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "foreground_iou": float(foreground), "background_iou": float(background),
            "miou": float((foreground + background) / 2)}


class RegionMetrics:
    def __init__(self, threshold=.5, threshold_start=.01, threshold_end=.99, threshold_step=.01):
        if not 0 <= threshold <= 1 or not 0 < threshold_start <= threshold_end < 1 or threshold_step <= 0:
            raise ValueError("Invalid metric thresholds")
        thresholds = np.arange(threshold_start, threshold_end + threshold_step * 1e-6, threshold_step)
        if len(thresholds) > 10001:
            raise ValueError("Threshold grid too large")
        self.thresholds = np.round(thresholds, 8)
        self.threshold = threshold
        self.counts = np.zeros(4, dtype=np.int64)  # TP FP FN TN
        self.sweep_tp = np.zeros(len(thresholds), dtype=np.int64)
        self.sweep_fp = np.zeros(len(thresholds), dtype=np.int64)
        self.positives, self.ois_sum, self.images = 0, 0.0, 0

    def update(self, probabilities, target):
        p, y = np.asarray(probabilities), np.asarray(target)
        if p.shape != y.shape or p.size == 0 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Metrics require matching, finite nonempty probability/target arrays")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Evaluation targets must be binary")
        p, y = p.reshape(-1), y.reshape(-1).astype(bool)
        predicted = p >= self.threshold
        tp, fp = np.count_nonzero(predicted & y), np.count_nonzero(predicted & ~y)
        positives = int(np.count_nonzero(y))
        fn, tn = positives - tp, y.size - positives - fp
        self.counts += [tp, fp, fn, tn]
        # Bucket each pixel once. Suffix sums give exact >= threshold counts.
        # Use the probability dtype for representable threshold comparisons.
        buckets = np.searchsorted(self.thresholds.astype(p.dtype), p, side="right")
        hp = np.bincount(buckets[y], minlength=len(self.thresholds) + 1)
        hn = np.bincount(buckets[~y], minlength=len(self.thresholds) + 1)
        sweep_tp, sweep_fp = np.cumsum(hp[::-1])[::-1][1:], np.cumsum(hn[::-1])[::-1][1:]
        f1s = divide(2 * sweep_tp, sweep_tp + sweep_fp + positives, empty=1.0)
        self.sweep_tp += sweep_tp
        self.sweep_fp += sweep_fp
        self.positives += positives
        self.ois_sum += float(f1s.max())
        self.images += 1
        return from_counts(int(tp), int(fp), int(fn), int(tn))

    def compute(self):
        if self.images == 0:
            raise ValueError("Cannot compute metrics for an empty dataset")
        tp, fp, fn, tn = map(int, self.counts)
        f1s = divide(2 * self.sweep_tp, self.sweep_tp + self.sweep_fp + self.positives, empty=1.0)
        best = int(np.argmax(f1s))
        return {**from_counts(tp, fp, fn, tn), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "threshold": self.threshold, "ods": float(f1s[best]), "ods_threshold": float(self.thresholds[best]),
                "ois": self.ois_sum / self.images, "images": self.images,
                "thresholds": self.thresholds.tolist(), "f1_by_threshold": f1s.tolist(),
                "empty_policy": "absent-class IoU=1; empty-vs-empty F1=1; undefined precision=1 only if target empty; undefined recall=1"}
