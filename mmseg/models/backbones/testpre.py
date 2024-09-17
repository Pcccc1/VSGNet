import torch
import torch.nn.functional as F
import torchvision.models as models
from mmcv import Config
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint


# dinov2_vitb14 = torch.hub.load(
#     "dinov2", "dinov2_vitb14", source="local", pretrained=False
#
# dinov2_vitb14.load_state_dict(torch.load("dinov2_vitb14_pretrain.pth"))
# dinov2_vitg14 = torch.hub.load(
#     "dinov2", "dinov2_vitg14", source="local", pretrained=False
# )
# dinov2_vitg14.load_state_dict(torch.load("dinov2_vitg14_pretrain.pth"))
# print(dinov2_vitb14)
# print(dinov2_vitg14)
# arr = torch.rand((4, 3, 280, 280))
# # arr1 = dinov2_vitb14(arr)
# # arr2 = dinov2_vitg14(arr)
# features = dinov2_vitb14.forward_features(arr)["x_norm_patchtokens"]
# print(features.shape)
# arr = arr.view(2, 6, 16, 32)
# out = F.interpolate(arr, size=(512, 512), mode="bilinear", align_corners=False)
# arr1 = arr1.reshape((-1, 32, 32, ))
# print(arr1.shape)
# print(arr2.shape)

# model = models.vit_b_16(pretrained=False).cpu()
checkpoint = torch.load("vit-b-checkpoint-1599.pth", map_location=torch.device("cpu"))

# state_dict = checkpoint["model"]
# print(checkpoint)
# model = build_model
