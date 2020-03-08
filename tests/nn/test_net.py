import torch

from nn.net import Net


def test_net():
    net = Net()
    result = net(torch.Tensor([0, 0, 0, 0, 0, 0]))
    assert result.shape == (3,)


def test_net_batch():
    net = Net()
    result = net(torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).view(2, 6))
    assert result.shape == (2, 3)