from compressai.entropy_models import EntropyModel
from compressai.ops import LowerBound
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


# The class don't do entropy coding, so no need to inherit 'EntropyModel'
class GaussianMixtureConditional(nn.Module):
    def __init__(self, N=192, K=3, scale_bound=0.11, likelihood_bound=1e-9):
        super().__init__()

        self.N = N
        self.K = K
        self.lower_bound_scale = LowerBound(scale_bound)
        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)
    
    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def _likelihood(self, inputs, scales_hat, means_hat, weights_hat):
        likelihood = torch.zeros_like(inputs)
        half = float(0.5)

        for k in range(self.K):
            values = inputs - means_hat[:, k]
            scales = self.lower_bound_scale(scales_hat[:, k])
            values = abs(values)

            upper = self._standardized_cumulative((half - values) / scales)
            lower = self._standardized_cumulative((-half - values) / scales)
            likelihood += (upper - lower) * weights_hat[:, k]

        return likelihood

    def quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        outputs = torch.round(outputs)

        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs

    def forward(self, inputs, scales, means, weights):
        outputs = self.quantize(inputs, 'noise' if self.training else 'dequantize', means=None)

        likelihood = self._likelihood(outputs, scales, means, weights)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)

        return outputs, likelihood
