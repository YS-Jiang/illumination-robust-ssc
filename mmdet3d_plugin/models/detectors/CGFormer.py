import torch
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS
from mmdet3d.models import builder

@DETECTORS.register_module()
class CGFormer(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        depth_net,
        img_view_transformer,
        proposal_layer,
        VoxFormer_head,
        occ_encoder_backbone=None,
        occ_encoder_neck=None,
        pts_bbox_head=None,
        depth_loss=False,
        use_rcf=False,
        train_cfg=None,
        test_cfg=None
    ):
        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        # img_neck may be None (e.g. DepthAnythingV2Backbone emits the final 640-ch map directly)
        self.img_neck = builder.build_neck(img_neck) if img_neck is not None else None

        # RCF: the grayscale image becomes a second value stream fused inside the 3D deformable
        # cross-attention with a per-voxel-query reliability weight.
        self.use_rcf = use_rcf

        self.depth_net = builder.build_neck(depth_net)
        if img_view_transformer is not None:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.VoxFormer_head = builder.build_head(VoxFormer_head)

        if occ_encoder_backbone is not None:
            self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        if occ_encoder_neck is not None:
            self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.depth_loss = depth_loss

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape   
        imgs = imgs.view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)

        if self.img_neck is not None:
            x = self.img_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]

        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x
    
    def extract_img_feat(self, img_inputs, img_metas):
        mlvl_feats_gray = None
        mlp_input = self.depth_net.get_mlp_input(*img_inputs[1:7])
        if self.use_rcf and len(img_inputs) > 8:
            # Color and grayscale stay separate: depth (geometry) comes from color, and the grayscale
            # features form a second context stream for the cross-attention fusion.
            color_img, gray_img = img_inputs[0], img_inputs[8]
            B = color_img.shape[0]
            both = self.image_encoder(torch.cat([color_img, gray_img], dim=0))
            color_feats, gray_feats = both[:B], both[B:]
            context, depth = self.depth_net([color_feats] + img_inputs[1:7] + [mlp_input], img_metas)
            context_gray, _ = self.depth_net([gray_feats] + img_inputs[1:7] + [mlp_input], img_metas)
            mlvl_feats_gray = [context_gray]
        else:
            img_enc_feats = self.image_encoder(img_inputs[0])
            context, depth = self.depth_net([img_enc_feats] + img_inputs[1:7] + [mlp_input], img_metas)

        if hasattr(self, 'img_view_transformer'):
            coarse_queries = self.img_view_transformer(context, depth, img_inputs[1:7])
        else:
            coarse_queries = None

        proposal = self.proposal_layer(img_inputs[1:7], img_metas)

        x = self.VoxFormer_head(
            [context],
            proposal,
            cam_params=img_inputs[1:7],
            lss_volume=coarse_queries,
            img_metas=img_metas,
            mlvl_dpt_dists=[depth.unsqueeze(1)],
            mlvl_feats_gray=mlvl_feats_gray,
        )

        return x, depth
    
    def occ_encoder(self, x):
        if hasattr(self, 'occ_encoder_backbone'):
            x = self.occ_encoder_backbone(x)
        
        if hasattr(self, 'occ_encoder_neck'):
            x = self.occ_encoder_neck(x)
        
        return x

    def _head_logits(self, img_inputs, img_metas, gt_occ):
        """Full image->voxel forward returning the occupancy logits (and depth)."""
        img_voxel_feats, depth = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)
        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc, img_metas=img_metas, img_feats=None, gt_occ=gt_occ)
        return output['output_voxels'], depth

    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        output_voxels, depth = self._head_logits(img_inputs, img_metas, gt_occ)

        losses = dict()

        if self.depth_loss and depth is not None:
            losses['loss_depth'] = self.depth_net.get_depth_loss(img_inputs['gt_depths'], depth)

        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output_voxels,
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        pred = output_voxels
        pred = torch.argmax(pred, dim=1)

        train_output = {
            'losses': losses,
            'pred': pred,
            'gt_occ': gt_occ
        }

        return train_output
    
    def forward_test(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        img_voxel_feats, depth = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)

        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]
        
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {
            'pred': pred,
            'gt_occ': gt_occ
        }

        return test_output

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)