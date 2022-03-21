from compressai.models.google import CompressionModel

import torch
import torch.nn as nn
from compressai.entropy_models import GaussianConditional



if __name__ == '__main__':
    net = Hyperprior().eval().cuda()

    with torch.no_grad():
        x = torch.rand((1, 3, 768, 512)).cuda()
        out = net(x)

    print(out['x_hat'].shape)
