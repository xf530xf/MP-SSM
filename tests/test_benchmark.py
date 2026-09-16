import pytest

from mp_ssm.benchmark import benchmark
from mp_ssm.config import load_config


def test_benchmark_reports_measurement_scope_and_partial_flops(tmp_path):
    config = load_config("configs/smoke.yaml")
    result = benchmark(config, tmp_path / "measured.json", sizes=[64], warmup=1, iterations=2)
    row = result["measurements"][0]
    assert row["latency_mean_ms"] > 0
    assert row["images_per_second"] == pytest.approx(1000 / row["latency_mean_ms"])
    assert row["partial_conv_linear_flops"] > 0
    assert row["memory"] is None
    assert "not total model FLOPs" in result["flops_scope"]
    assert result["serialized_state_dict_bytes"] > result["parameter_tensor_bytes"]


def test_large_reference_benchmark_requires_explicit_opt_in(tmp_path):
    config = load_config("configs/smoke.yaml")
    with pytest.raises(ValueError, match="Large reference scans"):
        benchmark(config, tmp_path / "large.json", sizes=[2048])
