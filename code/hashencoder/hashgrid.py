import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn


class HashEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2,
                 per_level_scale=2, base_resolution=16,
                 log2_hashmap_size=19, desired_resolution=None):
        super().__init__()

        # 与原版逻辑一致：如果给了 desired_resolution，则反推 per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(
                np.log2(desired_resolution / base_resolution) / (num_levels - 1)
            )

        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.per_level_scale = per_level_scale
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim

        self.encoder = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": float(per_level_scale),
                "interpolation": "Linear",
            },
        )

    def __repr__(self):
        return (f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} "
                f"level_dim={self.level_dim} base_resolution={self.base_resolution} "
                f"per_level_scale={self.per_level_scale}")

    def forward(self, inputs, size=1):
        # inputs: [..., input_dim], 归一化后的坐标，范围 [-size, size]
        # return: [..., num_levels * level_dim]

        inputs = inputs.detach()

        # 映射到 [0, 1]，与原版完全一致
        inputs = (inputs + size) / (2 * size)

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim).to(torch.float32)

        outputs = self.encoder(inputs)
        # tcnn 默认输出 fp16，转回 fp32 以兼容后续 MLP
        outputs = outputs.float()

        outputs = outputs.view(prefix_shape + [self.output_dim])
        return outputs