import torch

from nn.net import Net1Hidden, get_model_input


def test_net():
    net = Net1Hidden()
    result = net(torch.Tensor([0, 0, 0, 0, 0, 0]))
    assert result.shape == (3,)


def test_net_batch():
    net = Net1Hidden()
    result = net(torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).view(2, 6))
    assert result.shape == (2, 3)


def test_get_model_input():
    result = get_model_input(100, 200, 90, 190, 200, 300)
    assert torch.allclose(result, torch.Tensor([0.1250, 0.3333, 0.2500, 0.3750, 0.0125, 0.0167]), atol=0.0001)
    assert list(result.shape)[0] == 6
