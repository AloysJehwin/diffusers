# coding=utf-8
# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from diffusers import AutoencoderKLLTX2Video
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLLTX2VideoTesterConfig(BaseModelTesterConfig):
    @property
    def main_input_name(self):
        return "sample"

    @property
    def model_class(self):
        return AutoencoderKLLTX2Video

    @property
    def output_shape(self):
        return (3, 9, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 8,
            "block_out_channels": (8, 8, 8, 8),
            "decoder_block_out_channels": (16, 32, 64),
            "layers_per_block": (1, 1, 1, 1, 1),
            "decoder_layers_per_block": (1, 1, 1, 1),
            "spatio_temporal_scaling": (True, True, True, True),
            "decoder_spatio_temporal_scaling": (True, True, True),
            "decoder_inject_noise": (False, False, False, False),
            "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
            "upsample_residual": (True, True, True),
            "upsample_factor": (2, 2, 2),
            "timestep_conditioning": False,
            "patch_size": 1,
            "patch_size_t": 1,
            "encoder_causal": True,
            "decoder_causal": False,
            "encoder_spatial_padding_mode": "zeros",
            # Full model uses `reflect` but this does not have deterministic backward implementation, so use `zeros`
            "decoder_spatial_padding_mode": "zeros",
        }

    def get_dummy_inputs(self):
        batch_size = 2
        num_frames = 9
        num_channels = 3
        sizes = (16, 16)
        image = randn_tensor(
            (batch_size, num_channels, num_frames, *sizes), generator=self.generator, device=torch_device
        )
        return {"sample": image}


class TestAutoencoderKLLTX2Video(AutoencoderKLLTX2VideoTesterConfig, ModelTesterMixin):
    base_precision = 1e-2

    def test_outputs_equivalence(self):
        pytest.skip("Unsupported test.")


class TestAutoencoderKLLTX2VideoTraining(AutoencoderKLLTX2VideoTesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderKLLTX2Video."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "LTX2VideoEncoder3d",
            "LTX2VideoDecoder3d",
            "LTX2VideoDownBlock3D",
            "LTX2VideoMidBlock3d",
            "LTX2VideoUpBlock3d",
        }
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestAutoencoderKLLTX2VideoMemory(AutoencoderKLLTX2VideoTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderKLLTX2Video."""


class TestAutoencoderKLLTX2VideoSlicingTiling(AutoencoderKLLTX2VideoTesterConfig, AutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderKLLTX2Video."""


class TestAutoencoderKLLTX2VideoDecoderWidths(AutoencoderKLLTX2VideoTesterConfig):
    """The decoder must build coherent up-block widths for any decoder_block_out_channels.

    `conv_in` feeds an upsampler built for `out_channels * upscale_factor`, so both the
    decision to create it and the width it projects onto have to use that same number.
    Released LTX-2 checkpoints never take the branch, which is why this went unnoticed.
    """

    def _build(self, decoder_block_out_channels, upsample_factor):
        init_dict = self.get_init_dict()
        init_dict["decoder_block_out_channels"] = decoder_block_out_channels
        init_dict["upsample_factor"] = upsample_factor
        return AutoencoderKLLTX2Video(**init_dict).to(torch_device).eval()

    def test_matched_ratios_still_build_no_conv_in(self):
        # The shape released LTX-2 checkpoints use: every block's width divides down to
        # the next block's input, so conv_in is never needed and the weights are unchanged.
        model = self._build((16, 32, 64), (2, 2, 2))
        assert all(block.conv_in is None for block in model.decoder.up_blocks)

    @pytest.mark.parametrize(
        "decoder_block_out_channels,upsample_factor",
        [((16, 32, 48), (2, 1, 2)), ((24, 32, 40), (1, 2, 2)), ((16, 24, 64), (2, 2, 1))],
    )
    def test_uneven_widths_decode(self, decoder_block_out_channels, upsample_factor):
        model = self._build(decoder_block_out_channels, upsample_factor)
        assert any(block.conv_in is not None for block in model.decoder.up_blocks)

        latent = randn_tensor(
            (1, model.config.latent_channels, 3, 8, 8), device=torch.device(torch_device), dtype=torch.float32
        )
        with torch.no_grad():
            model.decode(latent)
