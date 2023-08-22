"""
Basic MobileViT-Track model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from .layers.neck_lighttrack import build_neck, build_feature_fusor

from .layers.head import build_box_head

from lib.models.mobilevit_track.mobilevit import MobileViT
from lib.utils.box_ops import box_xyxy_to_cxcywh
from easydict import EasyDict as edict

class MobileViT_Track(nn.Module):
    """ This is the base class for MobileViT-Track """

    def __init__(self, backbone, neck, feature_fusor, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        if neck is not None:
            self.neck = neck
            self.feature_fusor = feature_fusor
        if hasattr(self.backbone, "position_embeddings"):
            self.query_embed = nn.Embedding(16, feature_fusor.d_model)
            self.input_proj = nn.Conv2d(backbone.model_conf_dict['layer4']['out'], feature_fusor.d_model, kernel_size=1)
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor, search: torch.Tensor):
        x, z = self.backbone(x=search, z=template)

        # Forward neck
        x, z = self.neck(x, z)

        # add positional embeddings
        if hasattr(self.backbone, "position_embeddings"):
            x_pos = self.backbone.position_embeddings(x).to(x.dtype)
            z_pos = self.backbone.position_embeddings(z).to(z.dtype)
            feat_fused = self.feature_fusor(self.input_proj(z), self.input_proj(x), z_pos, x_pos, self.query_embed.weight)
        else:
            # Forward feature fusor
            feat_fused = self.feature_fusor(z, x)

        # Forward head
        out = self.forward_head(feat_fused, None)

        # out.update(aux_dict)
        # out['backbone_feat'] = [x, z]
        return out

    def forward_head(self, backbone_feature, gt_score_map=None):
        """
        backbone_feature: output embeddings of the backbone for search region
        """
        opt_feat = backbone_feature.contiguous()
        bs, _, _, _ = opt_feat.size()

        if self.head_type == "CORNER":
            # run the corner head
            pred_box = self.box_head(opt_feat, return_dist=False)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new
                   }
            return out

        elif self.head_type == "CENTER" or self.head_type == "CENTER_shared" or self.head_type == "CENTER_LITE" or \
                                                                                self.head_type == "CENTER_LITE_v2":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out

        elif self.head_type == "CENTER_MLP":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out

        elif self.head_type == "MLP":
            # reshape/flatten the neck output
            opt_feat = opt_feat.flatten(2).permute(0, 2, 1) # [B, C, W, H] -> [B, C, WH] -> [B, WH, C]
            # run the MLP head
            pred_box = self.box_head(opt_feat)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new
                   }
            return out

        elif self.head_type == "CORNER_LITE_REP_v2":
            # run the corner head
            pred_box = self.box_head(opt_feat, return_dist=False)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new
                   }
            return out


        else:
            raise NotImplementedError


def build_mobilevit_track(cfg, training=True):
    """
    function to create the hybrid-stream tracker with MobileViT backbone
    Args:
        cfg: EasyDict
        containing "DATA", "MODEL", "TRAIN", and "TEST" related info as keys
    Returns:
        model: instance of class MobileViT-Track
        containing (a) backbone: instance of class MobileViT
                   (b) box_head: for target score generation and bounding box regression
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')

    if cfg.MODEL.PRETRAIN_FILE and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'mobilevit_s':
        backbone = create_mobilevit_backbone(pretrained)
        hidden_dim = backbone.model_conf_dict['layer4']['out']
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'mobilevit_xs':
        backbone = create_mobilevit_backbone(pretrained)
        hidden_dim = backbone.model_conf_dict['layer4']['out']
        patch_start_index = 1
    else:
        raise NotImplementedError

    # finetune the backbone for the downstream task of object tracking
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # create positional embeddings
    if cfg.MODEL.NECK.TYPE == "FEATURE_FUSOR_HCAT":
        pos_embed_settings = edict()
        pos_embed_settings.position_embedding = cfg.MODEL.NECK.TYPE_EMBED
        pos_embed_settings.hidden_dim = 256
        backbone.position_embeddings = build_position_encoding(settings=pos_embed_settings)

    # build neck module to fuse template and search region features
    if cfg.MODEL.NECK:
        neck = build_neck(cfg=cfg, hidden_dim=hidden_dim)
        if cfg.MODEL.NECK.TYPE == 'BN_FEATURE_FUSOR_LIGHTTRACK':
            feature_fusor = build_feature_fusor(cfg=cfg, num_features=cfg.MODEL.NECK.NUM_CHANNS_POST_XCORR)
        elif cfg.MODEL.NECK.TYPE == 'FEATURE_FUSOR_HCAT':
            hcat_feature_fusor_settings = edict()
            hcat_feature_fusor_settings['hidden_dim'] = 256
            hcat_feature_fusor_settings['dropout'] = 0.1
            hcat_feature_fusor_settings['nheads'] = 8
            hcat_feature_fusor_settings['dim_feedforward'] = 2048
            hcat_feature_fusor_settings['featurefusion_layers'] = 2
            feature_fusor = build_featurefusion_network(hcat_feature_fusor_settings)
    else:
        neck = None

    # create the decoder module for target classification and bounding box regression
    box_head = build_box_head(cfg, cfg.MODEL.HEAD.NUM_CHANNELS)

    # create the MobileViT-based tracker
    model = MobileViT_Track(
        backbone,
        neck,
        feature_fusor,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    # To resume training from a previously saved checkpoint of MobileViT-Track
    if 'mobilevit_track' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)

        assert missing_keys == [] and unexpected_keys == [], "The backbone layers do not exactly match with the " \
                                                             "checkpoint state dictionaries. Please have a look at " \
                                                             "what those missing keys are!"

        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model


def create_mobilevit_backbone(pretrained):
    """
    function to create an instance of MobileViT backbone
    Args:
        pretrained:  str
        path to the pretrained image classification model to initialize the weights.
        If empty, the weights are randomly initialized
    Returns:
        model: nn.Module
        An object of Pytorch's nn.Module with MobileViT backbone (i.e., layer-1 to layer-4)
    """
    opts = {}
    opts['mode'] = 'small'
    opts['head_dim'] = None
    opts['number_heads'] = 4
    opts['conv_layer_normalization_name'] = 'batch_norm'
    opts['conv_layer_activation_name'] = 'relu'
    model = MobileViT(opts)

    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

        assert missing_keys == [], "The backbone layers do not exactly match with the checkpoint state dictionaries. " \
                                   "Please have a look at what those missing keys are!"

        print('Load pretrained model from: ' + pretrained)

    return model
