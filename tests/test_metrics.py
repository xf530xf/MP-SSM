import numpy as np
import pytest

from mp_ssm.metrics import RegionMetrics, from_counts


def test_hand_computed_confusion():
    m = RegionMetrics()
    m.update(np.array([.9, .8, .2, .1]), np.array([1, 0, 1, 0]))
    result = m.compute()
    assert [result[k] for k in ["tp", "fp", "fn", "tn"]] == [1, 1, 1, 1]
    assert result["f1"] == .5
    assert result["miou"] == pytest.approx(1/3)
    assert result["precision"] == .5


@pytest.mark.parametrize("p,y,miou,f1", [([0, 0], [0, 0], 1, 1), ([1, 1], [1, 1], 1, 1),
                                          ([0, 0], [1, 1], 0, 0), ([1, 1], [0, 0], 0, 0),
                                          ([0, 0], [1, 0], .25, 0)])
def test_empty_and_constant_cases(p, y, miou, f1):
    metric = RegionMetrics()
    metric.update(np.array(p, dtype=np.float32), np.array(y))
    result = metric.compute()
    assert result["miou"] == miou
    assert result["f1"] == f1
    assert all(np.isfinite(result[key]) for key in ("precision", "recall", "f1", "miou", "ods", "ois"))


def test_streaming_ods_ois_matches_bruteforce():
    random = np.random.default_rng(5)
    probabilities = [random.random((5, 7)).astype(np.float32) for _ in range(3)]
    targets = [random.integers(0, 2, size=(5, 7)) for _ in range(3)]
    probabilities[0][0, :3] = [.01, .5, .99]  # exactly representable comparisons at thresholds
    metric = RegionMetrics()
    for probability, target in zip(probabilities, targets):
        metric.update(probability, target)
    all_f1, pooled = [], []
    for threshold in metric.thresholds:
        counts, image_f1 = np.zeros(4, dtype=np.int64), []
        for probability, target in zip(probabilities, targets):
            pred, truth = probability >= np.float32(threshold), target.astype(bool)
            tp, fp = (pred & truth).sum(), (pred & ~truth).sum()
            fn, tn = (~pred & truth).sum(), (~pred & ~truth).sum()
            counts += [tp, fp, fn, tn]
            image_f1.append(from_counts(tp, fp, fn, tn)["f1"])
        all_f1.append(image_f1)
        pooled.append(from_counts(*counts)["f1"])
    result = metric.compute()
    assert result["ods"] == pytest.approx(max(pooled))
    assert result["ois"] == pytest.approx(np.max(all_f1, axis=0).mean())
    np.testing.assert_allclose(result["f1_by_threshold"], pooled)


def test_miou_is_two_class_average():
    result = from_counts(1, 0, 1, 8)
    assert result["foreground_iou"] == .5
    assert result["miou"] == pytest.approx((.5 + 8/9)/2)
