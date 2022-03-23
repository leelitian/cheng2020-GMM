import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from entropy_model import GaussianMixtureConditional
from compressai.models import CompressionModel
from range_coder import RangeEncoder, RangeDecoder

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    MaskedConv2d,
    conv3x3,
    subpel_conv3x3,
)


class EntropyParameters(nn.Module):
    def __init__(self, N=192, K=3) -> None:
        super().__init__()

        self.N = N
        self.K = K
        self.layers = nn.Sequential(
            nn.Conv2d(N * 12 // 3, N * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 10 // 3, N * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 10 // 3, 3 * N * K, 1),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class Cheng2020GMM(CompressionModel):
    def __init__(self, N=192, K=3, **kwargs):
        super().__init__(N, **kwargs)

        self.N = N
        self.K = K

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.context_prediction = MaskedConv2d(
            N, 2 * N, kernel_size=5, padding=2, stride=1
        )

        self.entropy_parameters = EntropyParameters(N=N, K=K)
        self.gaussian_conditional = GaussianMixtureConditional()

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        # just round, no need to minus means
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize", means=None
        )

        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat, weights_hat = gaussian_params.chunk(3, 1)

        B, C, H, W = y_hat.shape
        scales_hat = scales_hat.view(B, self.K, C, H, W)
        means_hat = means_hat.view(B, self.K, C, H, W)
        weights_hat = weights_hat.view(B, self.K, C, H, W)
        weights_hat = F.softmax(weights_hat, dim=1)

        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means_hat, weights_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # range-coder is used to coding y_hat
    def compress(self, x, stream_name):
        """
        https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/network.py
        When batch_size is 1
        """
        torch.backends.cudnn.deterministic = True

        # path to save codestreams
        z_stream_path = os.path.join(stream_name + '.npz')
        y_stream_path = os.path.join(stream_name + '.bin')

        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat)

        # just round, no need to minus means
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize", means=None
        )

        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat, weights_hat = gaussian_params.chunk(3, 1)

        B, C, H, W = y_hat.shape
        scales_hat = scales_hat.view(B, self.K, C, H, W)
        means_hat = means_hat.view(B, self.K, C, H, W)
        weights_hat = weights_hat.view(B, self.K, C, H, W)
        weights_hat = F.softmax(weights_hat, dim=1)

        # compute the zero channel and abs'max
        y_hat_np = y_hat.cpu().numpy().astype('int')
        flag = np.zeros(C, dtype=np.int32)
        for ch in range(C):
            if np.sum(abs(y_hat_np[:, ch, :, :])) > 0:
                flag[ch] = 1

        non_zero_idx = np.squeeze(np.where(flag == 1))
        flag_nums = np.packbits(flag.reshape([8, C // 8]))
        minmax = np.maximum(abs(y_hat_np.max()), abs(y_hat_np.min()))
        minmax = int(np.maximum(minmax, 1))

        # write z
        fileobj = open(z_stream_path, mode="wb")
        # x.H, x.W
        fileobj.write(np.array(x.shape[2:], dtype=np.uint16).tobytes())
        fileobj.write(np.array([len(z_strings[0]), minmax], dtype=np.uint16).tobytes())
        fileobj.write(np.array(flag_nums, dtype=np.uint8).tobytes())
        fileobj.write(z_strings[0])
        fileobj.close()

        # compress y_hat
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2
        encoder = RangeEncoder(y_stream_path)
        samples = torch.arange(0, minmax * 2 + 1).float().to(x.device)
        y_hat = F.pad(y_hat, (padding, padding, padding, padding))

        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in tqdm(range(H)):
            for w in range(W):
                y_crop = y_hat[0: 1, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    weight=masked_weight,
                    bias=self.context_prediction.bias
                )
                p = params[0: 1, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_crop, means_crop, weights_crop = gaussian_params.chunk(3, 1)     # (1, 3C, H, W)
                scales_crop = scales_crop.view(self.K, self.N)                          # (1, C, H, W)
                means_crop = means_crop.view(self.K, self.N) + minmax
                weights_crop = weights_crop.view(self.K, self.N)
                weights_crop = F.softmax(weights_crop, dim=0)

                for i in range(len(non_zero_idx)):
                    ch = non_zero_idx[i]
                    scale_p = scales_crop[:, ch]
                    mean_p = means_crop[:, ch]
                    weight_p = weights_crop[:, ch]

                    pmf = torch.zeros_like(samples)
                    for k in range(self.K):
                        half = float(0.5)
                        value = abs(samples - mean_p[k])
                        scale = self.gaussian_conditional.lower_bound_scale(scale_p[k])
                        upper = self.gaussian_conditional._standardized_cumulative((half - value) / scale)
                        lower = self.gaussian_conditional._standardized_cumulative((-half - value) / scale)
                        pmf += (upper - lower) * weight_p[k]

                    pmf = pmf.cpu().numpy()
                    pmf_clip = np.clip(pmf, 1.0 / 65536, 1.0)
                    pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                    cdf = list(np.add.accumulate(pmf_clip))
                    cdf = [0] + [int(j) for j in cdf]
                    symbol = int(y_hat_np[0, ch, h, w] + minmax)
                    encoder.encode([symbol], cdf)
                    y_hat[0, ch, h + padding, w + padding] = symbol - minmax

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        encoder.close()
        torch.backends.cudnn.deterministic = False

        return {
            "y_stream": y_stream_path,
            "z_stream": z_stream_path,
        }

    def decompress(self, stream_name):
        """
        https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/network.py
        When batch_size is 1
        """
        torch.backends.cudnn.deterministic = True

        # path to save codestreams
        z_stream_path = os.path.join(stream_name + '.npz')
        y_stream_path = os.path.join(stream_name + '.bin')

        fileobj = open(z_stream_path, mode='rb')
        x_shape = np.frombuffer(fileobj.read(4), dtype=np.uint16)
        z_length, minmax = np.frombuffer(fileobj.read(4), dtype=np.uint16)
        flag_nums = np.frombuffer(fileobj.read(self.N // 8), dtype=np.uint8)
        z_strings = fileobj.read(z_length)
        fileobj.close()

        flag = np.unpackbits(flag_nums)
        non_zero_idx = np.squeeze(np.where(flag == 1))
        y_shape = x_shape // 16
        z_shape = y_shape // 4
        H, W = y_shape

        z_hat = self.entropy_bottleneck.decompress([z_strings], z_shape)
        params = self.h_s(z_hat)

        kernel_size = 5
        padding = (kernel_size - 1) // 2
        decoder = RangeDecoder(y_stream_path)
        samples = torch.arange(0, minmax * 2 + 1).float().to(z_hat.device)

        y_hat = torch.zeros(
            (1, self.N, H + 2 * padding, W + 2 * padding),
            device=z_hat.device,
        )

        for h in tqdm(range(H)):
            for w in range(W):
                y_crop = y_hat[0: 1, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    weight=self.context_prediction.weight,
                    bias=self.context_prediction.bias
                )

                p = params[0: 1, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_crop, means_crop, weights_crop = gaussian_params.chunk(3, 1)
                scales_crop = scales_crop.reshape(self.K, self.N)
                means_crop = means_crop.reshape(self.K, self.N) + minmax
                weights_crop = weights_crop.reshape(self.K, self.N)
                weights_crop = F.softmax(weights_crop, dim=0)

                for i in range(len(non_zero_idx)):
                    c = non_zero_idx[i]
                    scale_p = scales_crop[:, c]
                    mean_p = means_crop[:, c]
                    weight_p = weights_crop[:, c]

                    pmf = torch.zeros_like(samples)
                    for k in range(self.K):
                        half = float(0.5)
                        value = abs(samples - mean_p[k])
                        scale = self.gaussian_conditional.lower_bound_scale(scale_p[k])
                        upper = self.gaussian_conditional._standardized_cumulative((half - value) / scale)
                        lower = self.gaussian_conditional._standardized_cumulative((-half - value) / scale)
                        pmf += (upper - lower) * weight_p[k]

                    pmf = pmf.cpu().numpy()
                    pmf_clip = np.clip(pmf, 1.0/65536, 1.0)
                    pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                    cdf = list(np.add.accumulate(pmf_clip))
                    cdf = [0] + [int(j) for j in cdf]
                    symbol = decoder.decode(1, cdf)[0]
                    y_hat[0, c, h + padding, w + padding] = symbol - minmax

        decoder.close()
        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp(0, 1)
        torch.backends.cudnn.deterministic = False

        return {'x_hat': x_hat}


if __name__ == '__main__':
    net = Cheng2020GMM().eval().cuda()
    net.update(force=True)

    with torch.no_grad():
        x = torch.rand((1, 3, 768, 512)).cuda()
        # net.compress(x, 'foo')
        out = net.decompress('foo')

    print(out['x_hat'].shape)
