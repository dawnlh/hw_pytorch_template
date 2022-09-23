import torch
import torch.nn as nn
import torch.nn.functional as F
# ===========================
# global loss info extract
# ===========================
LOSSES = {}


def add2loss(cls):
    if cls.__name__ in LOSSES:
        raise ValueError(f'{cls.__name__} is already in the LOSSES list')
    else:
        LOSSES[cls.__name__] = cls
    return cls

# ===========================
# weighted_loss
# ===========================


class WeightedLoss(nn.Module):
    """
    weighted multi-loss
    loss_conf_dict: {loss_type1: weight | [weight,args_dict], ...}
    """

    def __init__(self, loss_conf_dict):
        super(WeightedLoss, self).__init__()
        self.loss_conf_dict = loss_conf_dict

        # instantiate classes
        self.losses = []
        for k, v in loss_conf_dict.items():
            if isinstance(v, (float, int)):
                assert v >= 0, f"loss'weight {k}:{v} should be positive"
                self.losses.append({'cls': LOSSES[k](), 'weight': v})
            elif isinstance(v, list) and len(v) == 2:
                assert v[0] >= 0, f"loss'weight {k}:{v} should be positive"
                self.losses.append({'cls': LOSSES[k](**v[1]), 'weight': v[0]})
            else:
                raise ValueError(
                    f"the Key({k})'s Value {v} in Dict(loss_conf_dict) should be scalar(weight) | list[weight, args] ")

    def forward(self, output, target):
        print(LOSSES)
        loss_v = 0
        for loss in self.losses:
            loss_v += loss['cls'](output, target)*loss['weight']

        return loss_v

# ===========================
# basic_loss
# ===========================


@add2loss
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        diff = output.to('cuda:0') - target.to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


@add2loss
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, output, target):
        loss = self.loss(self.laplacian_kernel(output.to('cuda:0')),
                         self.laplacian_kernel(target.to('cuda:0')))
        return loss


@add2loss
class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, output, target):
        diff = torch.fft.fft2(output.to('cuda:0')) - \
            torch.fft.fft2(target.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss


if __name__ == "__main__":
    import PerceptualLoss
    output = torch.randn(4, 3, 10, 10)
    target = torch.randn(4, 3, 10, 10)
    loss_conf_dict = {'CharbonnierLoss': 0.5, 'fftLoss': 0.5}

    Weighted_Loss = WeightedLoss(loss_conf_dict)
    loss_v = Weighted_Loss(output, target)
    print('loss: ', loss_v)
