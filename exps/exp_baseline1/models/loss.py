import torch
import torch.nn as nn


class mIoULoss(nn.Module):
    def __init__(self, reduce=True):
        super(mIoULoss, self).__init__()
        self.reduce = reduce

    def forward(self, inputs, target):
        inter = (inputs * target).sum(-1)
        union = inputs.sum(-1) + target.sum(-1) - inter
        IoULoss = 1 - inter / (union + 1e-8)
        if self.reduce:
            IoULoss = torch.mean(IoULoss)
        return IoULoss


if __name__ == '__main__':
    loss_fn = mIoULoss()
    inputs = torch.rand((16, 2048))
    target = torch.rand((16, 2048))
    loss = loss_fn(inputs, target)
    print(loss)
