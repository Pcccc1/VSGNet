import torch.nn as nn
import torch.nn.functional as F

from .init_func import init_weight

from ..utils.engine.logger import get_logger
from .VSGNet import VSGNet as backbone
from .MLPDecoder import DecoderHead

logger = get_logger()


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        cfg=None,
        criterion=nn.CrossEntropyLoss(reduction="mean", ignore_index=255),
        norm_layer=nn.BatchNorm2d,
        single_GPU=False,
    ):
        super(EncoderDecoder, self).__init__()
        self.norm_layer = norm_layer


        self.channels = [96, 192, 288, 576]

        if single_GPU:
            print("single GPU")
            norm_cfg = dict(type="BN", requires_grad=True)
        else:
            norm_cfg = dict(type="SyncBN", requires_grad=True)

        if cfg.drop_path_rate is not None:
            self.backbone = backbone(
                drop_path_rate=cfg.drop_path_rate, norm_cfg=norm_cfg
            )
        else:
            self.backbone = backbone(drop_path_rate=0.1, norm_cfg=norm_cfg)

        self.aux_head = None

        self.decode_head = DecoderHead(
            in_channels=self.channels,
            num_classes=cfg.num_classes,
            norm_layer=norm_layer,
            embed_dim=cfg.decoder_embed_dim,
        )

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=None)  # not use pretrain

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info("Loading pretrained model: {}".format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info("Initing weights ...")
        init_weight(
            self.decode_head,
            nn.init.kaiming_normal_,
            self.norm_layer,
            cfg.bn_eps,
            cfg.bn_momentum,
            mode="fan_in",
            nonlinearity="relu",
        )

    def encode_decode(self, rgb, modal_x):
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
        out = self.decode_head.forward(x)
        out = F.interpolate(
            out, size=orisize[-2:], mode="bilinear", align_corners=False
        )
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(
                aux_fm, size=orisize[2:], mode="bilinear", align_corners=False
            )
            return out, aux_fm
        return out

    def forward(self, rgb, modal_x=None, label=None):
        # print('builder',rgb.shape,modal_x.shape)
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)
        if label is not None:
            loss = self.criterion(out, label.long())
            return loss
        return out
