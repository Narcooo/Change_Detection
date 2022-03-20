import torch
from typing import Optional, Union, List
from segmentation_models_pytorch.encoders import get_encoder
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch import unet
# from .dual_change import CSDecoder_dual


class CSDecoder(UnetDecoder):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            center=True,
            attention_type=None,
    ):

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
        encode_unionchannel = []

        for ec in encoder_channels:
            encode_unionchannel.append(ec * 2)

        super().__init__(
            encoder_channels=encode_unionchannel,
            decoder_channels=decoder_channels,
            n_blocks=n_blocks,
            use_batchnorm=use_batchnorm,
            center=center,
            attention_type=attention_type,
        )

        block_head = [
            SegmentationHead(in_channels=in_ch, out_channels=1, activation='sigmoid', kernel_size=3) for in_ch in
            decoder_channels[0:]
        ]
        self.blocks_head = nn.ModuleList(block_head)
        self.initialize()

    def initialize(self):
        for i, decoder_block in enumerate(self.blocks_head):
            init.initialize_decoder(self.blocks_head[i])

    def forward(self, features0, features1):
        features = []
        for i in range(len(features0)):
            features.append(torch.cat([features0[i], features1[i]], 1))

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]
        head = features[0]
        skips = features[1:]

        x = self.center(head)
        all = []
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            all.append(self.blocks_head[i](x))

        if self.training:
            return all
        else:
            return self.blocks_head[-1](x)


class ChangeModel_CAD(torch.nn.Module):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: str = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            in_channels: int = 3,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.change_decoder = CSDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
        )
        self.change_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            activation='sigmoid',
            kernel_size=3,
        )
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.change_decoder)
        init.initialize_decoder(self.change_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features0 = self.encoder(x[:, :3, :, :])
        features1 = self.encoder(x[:, 3:, :, :])

        fe_cut = []
        fe_add = []

        # change attention
        for fe in zip(features0, features1):
            fe0, fe1 = fe
            feindex_add = (fe0 + fe1)/2.0
            feindex_cut = fe0 - fe1

            fe_add.append(feindex_add)
            fe_cut.append(feindex_cut)

        change_output = self.change_decoder(fe_cut, fe_add)

        return change_output


if __name__ == '__main__':
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')

    device = torch.device("cuda:0" if train_on_gpu else "cpu")

    ENCODER = "efficientnet-b1"
    ENCODER_WEIGHTS = None#"imagenet"
    ACTIVATION = None

    model = ChangeModel_CAD(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
    )

    img = np.zeros((1, 6, 512, 512), 'float32')
    img = torch.from_numpy(img)

    change_output = model(img)
    print(change_output.shape)

