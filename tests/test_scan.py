import math

import pytest
import torch

from mp_ssm.scan import SelectiveScan2D, require_cuda_scan, scan_indices, selective_scan_reference


@pytest.mark.parametrize("height,width", [(1, 1), (1, 5), (7, 1), (3, 5), (5, 3), (8, 8)])
def test_all_routes_are_invertible(height, width):
    routes, inverse = scan_indices(height, width)
    expected = torch.arange(height * width)
    assert routes.shape == (6, height * width)
    for route, inv in zip(routes, inverse):
        assert torch.equal(route.sort().values, expected)
        assert torch.equal(route[inv], expected)
    if height > 1 and width > 1:
        assert not torch.equal(routes[4], routes[0])
        assert not torch.equal(routes[5], routes[2])


def test_scalar_recurrence_and_analytic_gradient():
    u = torch.tensor([[[1., 2., 3.]]], requires_grad=True)
    y = selective_scan_reference(u, torch.zeros_like(u), torch.tensor([[-1.]]),
                                  torch.ones(1, 1, 1, 3), torch.ones(1, 1, 1, 3), torch.zeros(1))
    expected = torch.tensor([[[1., 2.5, 4.25]]]) * math.log(2)
    torch.testing.assert_close(y, expected)
    y.sum().backward()
    torch.testing.assert_close(u.grad, torch.tensor([[[1.75, 1.5, 1.]]]) * math.log(2))


def test_selective_parameters_receive_gradients():
    scan = SelectiveScan2D(4, d_state=3, expand=1, backend="reference")
    x = torch.randn(2, 4, 3, 5, requires_grad=True)
    output = scan(x)
    assert output.shape == x.shape
    output.square().mean().backward()
    for name, parameter in scan.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.cuda
def test_cuda_reference_forward_and_backward():
    if not torch.cuda.is_available():
        pytest.skip("No NVIDIA CUDA device")
    try:
        require_cuda_scan()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    reference = SelectiveScan2D(4, d_state=3, expand=2, backend="reference").cuda()
    cuda = SelectiveScan2D(4, d_state=3, expand=2, backend="cuda").cuda()
    cuda.load_state_dict(reference.state_dict())
    x1 = torch.randn(2, 4, 3, 5, device="cuda", requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, y2 = reference(x1), cuda(x2)
    torch.testing.assert_close(y1, y2, atol=2e-5, rtol=2e-4)
    y1.square().sum().backward()
    y2.square().sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad, atol=1e-4, rtol=1e-3)
    for first, second in zip(reference.parameters(), cuda.parameters()):
        torch.testing.assert_close(first.grad, second.grad, atol=2e-4, rtol=2e-3)


def test_no_silent_cuda_fallback():
    with pytest.raises(RuntimeError, match="CUDA tensor"):
        SelectiveScan2D(4, backend="cuda")(torch.randn(1, 4, 3, 5))


def test_cached_indices_can_transition_from_inference_to_training():
    scan_indices.cache_clear()
    first = SelectiveScan2D(4, d_state=2, expand=1, backend="reference")
    x = torch.randn(1, 4, 4, 7)
    with torch.inference_mode():
        first(x)
    # Reuse both the module device cache and the cross-module CPU cache.
    second = SelectiveScan2D(4, d_state=2, expand=1, backend="reference")
    for model in (first, second):
        y = model(x.requires_grad_())
        y.square().sum().backward()
        assert model.x_proj.grad.isfinite().all()
