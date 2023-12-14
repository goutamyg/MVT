"""
Basic MobileViT-Track model.
"""
import importlib
import os
import numpy as np
import torch
from torch import nn

from lib.models.mobilevit_track.layers.neck_lighttrack import build_neck, build_feature_fusor
from lib.models.mobilevit_track.layers.head import build_box_head
from torch.nn.modules.transformer import _get_clones

from lib.models.mobilevit_track.mobilevit import MobileViT
from lib.test.evaluation.environment import env_settings
from lib.test.evaluation import Tracker

from numerize import numerize

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

        # Forward feature fusor
        feat_fused = self.feature_fusor(z, x)

        # Forward head
        out = self.forward_head(feat_fused, None)

        return out

    def forward_head(self, backbone_feature, gt_score_map=None):
        """
        backbone_feature: output embeddings of the backbone for search region
        """
        opt_feat = backbone_feature.contiguous()
        bs, _, _, _ = opt_feat.size()

        if self.head_type == "CENTER":
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
        else:
            raise NotImplementedError

def build_mobilevit_track(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    # build mobilevit backbone
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

    # build neck module to fuse template and search region features
    if cfg.MODEL.NECK:
        neck = build_neck(cfg=cfg, hidden_dim=hidden_dim)
        if cfg.MODEL.NECK.TYPE == 'BN_FEATURE_FUSOR_LIGHTTRACK':
            feature_fusor = build_feature_fusor(cfg=cfg, num_features=cfg.MODEL.NECK.NUM_CHANNS_POST_XCORR)
    else:
        neck = None

    # create the decoder module for target classification and bounding box regression
    box_head = build_box_head(cfg, cfg.MODEL.HEAD.NUM_CHANNELS)

    model = MobileViT_Track(
        backbone=backbone,
        neck=neck,
        feature_fusor=feature_fusor,
        box_head=box_head,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

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


def get_parameters(name, parameter_name):
    """Get parameters."""
    param_module = importlib.import_module('lib.test.parameter.{}'.format(name))
    params = param_module.parameters(parameter_name)
    return params


if __name__ == "__main__":

    # convert model to onnx
    tracker_name = "mobilevit_track"
    tracker_param = "mobilevit_256_64x2_got10k_ep100_cosine_annealing"
    tracker = Tracker(tracker_name, tracker_param, "video")

    settings = env_settings()
    params = get_parameters(tracker_name, tracker_param)

    network = build_mobilevit_track(params.cfg, training=False)
    # ckpt = torch.load(params.checkpoint, map_location='cpu')["net"]
    network.load_state_dict(torch.load(params.checkpoint, map_location='cpu')['net'], strict=True)

    tracker_model_param_count = {"backbone":0, "neck":0, "feature_fusor":0, "box_head":0}

    # Count the model parameters
    for k, v in network.state_dict().items():
        print(
            "Processing variable: " + k + " with shape: ",
            v.shape,
            " and type: ",
            v.dtype,
        )
        for module in tracker_model_param_count:
            if module in k:
                tracker_model_param_count[module] += np.prod(v.shape)

    # Print the per-module and overall weights
    overall_param_count = 0
    for k, v in tracker_model_param_count.items():
        print("#params in {} = {}".format(k, numerize.numerize(int(v))))
        overall_param_count += v
    print("#params in total = {}".format(numerize.numerize(int(overall_param_count))))