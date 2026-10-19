import pytest
import torch

from tacos.model import TacosNet

pytestmark = pytest.mark.usefixtures("stub_encoder")


def make_net(context="all"):
    torch.manual_seed(0)
    return TacosNet(context=context)


def test_output_has_the_input_resolution():
    assert make_net()(torch.rand(2, 3, 64, 96), torch.rand(2, 4)).shape == (2, 1, 64, 96)


def test_no_context_is_a_plain_dpt():
    net = make_net("none")
    assert not any("context" in name for name, _ in net.named_parameters())
    assert len(net.configure_optimizers()["optimizer"].param_groups) == 1


def test_closed_gates_ignore_the_context():
    net, x = make_net().eval(), torch.rand(1, 3, 64, 96)
    assert torch.equal(net(x, torch.zeros(1, 4)), net(x, torch.full((1, 4), 5.0)))


def test_nan_context_is_read_as_the_train_mean():
    net, x = make_net(), torch.rand(1, 3, 64, 96)
    net.context_mean.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    for p in net.context_attention.parameters():          # open the gates so the context matters
        torch.nn.init.normal_(p)
    out = net(x, torch.tensor([[float("nan"), 0.2, 0.3, 0.4]]))
    assert torch.isfinite(out).all() and torch.equal(out, net(x, net.context_mean[None]))


def test_loss_ignores_unlabelled_pixels():
    target = torch.full((1, 1, 4, 4), float("nan"))
    target[0, 0, 0, 0] = 1.0
    expected = torch.nn.functional.huber_loss(torch.tensor([0.0]), torch.tensor([1.0]), delta=0.5)
    assert torch.isclose(make_net().loss(torch.zeros(1, 1, 4, 4), target), expected)


def test_loss_without_labels_is_zero_and_differentiable():
    loss = make_net().loss(torch.zeros(1, 1, 4, 4, requires_grad=True), torch.full((1, 1, 4, 4), float("nan")))
    assert loss.item() == 0.0 and loss.requires_grad


def test_context_attention_learns_ten_times_faster():
    groups = make_net().configure_optimizers()["optimizer"].param_groups
    assert sorted(g["initial_lr"] for g in groups) == pytest.approx([3e-5, 3e-4])
    assert all(g["weight_decay"] == 0.01 for g in groups)


def test_checkpoint_leaves_out_the_frozen_encoder():
    assert not any(k.startswith("encoder.") for k in make_net().state_dict())
