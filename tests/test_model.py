import pytest
import torch

from mp_ssm.model import MPSSM, MultiScaleFusion


@pytest.mark.parametrize("shape", [(32, 32), (33, 49), (64, 96)])
def test_original_resolution_and_gradient(shape):
    model = MPSSM(channels=[4, 8, 12, 16], fusion_channels=8, d_state=2, expand=1, scan_backend="reference")
    model.eval()
    x = torch.randn(1, 3, *shape, requires_grad=True)
    output = model(x)
    assert output.shape == (1, 1, *shape)
    assert output.isfinite().all()
    torch.nn.functional.binary_cross_entropy_with_logits(output, torch.zeros_like(output)).backward()
    assert x.grad.isfinite().all()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())


def test_multiscale_alignment_and_all_branch_gradients():
    model = MultiScaleFusion([12, 8, 4], 4, 2, 1, "reference")
    features = [torch.randn(1, c, h, w, requires_grad=True)
                for c, h, w in [(12, 2, 3), (8, 3, 5), (4, 5, 7)]]
    output = model(features)
    assert output.shape == (1, 4, 5, 7)
    output.square().mean().backward()
    assert all(x.grad.isfinite().all() and x.grad.abs().sum() > 0 for x in features)


def test_full_default_architecture_forward():
    model = MPSSM(scan_backend="reference").eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 64))
    assert output.shape == (1, 1, 64, 64)
    assert output.isfinite().all()
