"""
DepthAnythingV2Backbone: Depth Anything V2 (ViT-L) image backbone.
Emits a single 640-channel feature map at stride /8 ([B*N, 640, 48, 160] for a 384x1280 input),
so the downstream depth net, view transformer and heads are unchanged. Set img_neck=None.

input [B*N,3,384,1280] -> resize to the /14 grid (378x1274) -> ViT-L intermediate layers
[4,11,17,23] -> reshape tokens to [B*N,1024,27,91] -> fuse the 4 stages + Conv1x1 -> 640 ch
-> interpolate to 48x160. The ViT is frozen by default (only the adapter is trained).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import BACKBONES

# make the cloned Depth-Anything-V2 repository importable (override with the DAV2_ROOT env var)
_DAV2_ROOT = os.environ.get("DAV2_ROOT", "./packages/Depth-Anything-V2")
import sys
if _DAV2_ROOT not in sys.path:
    sys.path.insert(0, _DAV2_ROOT)


@BACKBONES.register_module()
class DepthAnythingV2Backbone(nn.Module):
    # ViT-L config for DepthAnythingV2
    _VIT_CFG = dict(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    _EMBED_DIM = 1024
    _LAYERS = [4, 11, 17, 23]   # 4 intermediate blocks (of 24) to tap, DPT-style
    _PATCH = 14

    def __init__(self,
                 out_channels=640,
                 out_stride=8,
                 input_hw=(384, 1280),
                 pretrained=os.path.join(_DAV2_ROOT, "checkpoints/depth_anything_v2_vitl.pth"),
                 freeze_vit=True,
                 **kwargs):
        super().__init__()
        from depth_anything_v2.dpt import DepthAnythingV2
        full = DepthAnythingV2(**self._VIT_CFG)
        if pretrained and os.path.isfile(pretrained):
            sd = torch.load(pretrained, map_location="cpu")
            full.load_state_dict(sd, strict=False)
            print(f"[DepthAnythingV2Backbone] loaded weights from {pretrained}")
        self.vit = full.pretrained                       # the DINOv2 ViT encoder
        self.freeze_vit = freeze_vit
        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()

        self.out_channels = out_channels
        self.out_stride = out_stride
        self.input_hw = input_hw
        # /14 grid nearest to input (cache; recomputed in forward from actual H,W)
        # adapter: fuse 4 token-stages (concat -> 1x1) to out_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(self._EMBED_DIM * len(self._LAYERS), out_channels, kernel_size=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_vit:
            self.vit.eval()   # keep ViT in eval (frozen BN/attn dropout off)
        return self

    def _grid14(self, h, w):
        # nearest smaller multiple of 14
        gh = max(self._PATCH, (h // self._PATCH) * self._PATCH)
        gw = max(self._PATCH, (w // self._PATCH) * self._PATCH)
        return gh, gw

    def forward(self, imgs):
        # imgs: [BN, 3, H, W]
        BN, _, H, W = imgs.shape
        gh, gw = self._grid14(H, W)
        x = F.interpolate(imgs, size=(gh, gw), mode="bilinear", align_corners=False)
        ph, pw = gh // self._PATCH, gw // self._PATCH   # token grid

        ctx = torch.no_grad() if self.freeze_vit else _nullcontext()
        with ctx:
            toks = self.vit.get_intermediate_layers(x, self._LAYERS, return_class_token=False)
        # each: [BN, ph*pw, 1024] -> [BN, 1024, ph, pw]
        maps = [t.permute(0, 2, 1).reshape(BN, self._EMBED_DIM, ph, pw).contiguous() for t in toks]
        feat = torch.cat(maps, dim=1)                    # [BN, 4*1024, ph, pw]
        feat = self.fuse(feat)                           # [BN, 640, ph, pw]

        out_h = H // self.out_stride                     # 48 for 384
        out_w = W // self.out_stride                     # 160 for 1280
        feat = F.interpolate(feat, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return (feat,)                                   # tuple so image_encoder's x[0] works


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *a): return False
