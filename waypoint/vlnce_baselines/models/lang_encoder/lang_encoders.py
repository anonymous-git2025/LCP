import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.models.lang_encoder.vlnbert.vlnbert_init import get_vlnbert_models
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)
from vlnce_baselines.models.policy import ILPolicy

from waypoint_prediction.utils import nms
from vlnce_baselines.models.utils import (
    angle_feature_with_ele, dir_angle_feature_with_ele, length2mask)
import math


class Lang_by_VLN_BERT(nn.Module):
    def __init__(
        self,
        model_config: Config, 
    ):
        super().__init__()

        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        print('\nInitalizing the Lang-by-VLN-BERT model ...')
        self.vln_bert_lang = get_vlnbert_models(config=None)
        self.vln_bert_lang.config.directions = 1 
        layer_norm_eps = self.vln_bert_lang.config.layer_norm_eps


      
        self.space_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(start_dim=2),)
        self.rgb_linear = nn.Sequential(
            nn.Linear(
                model_config.RGB_ENCODER.encode_size,
                model_config.RGB_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Linear(
                model_config.DEPTH_ENCODER.encode_size,
                model_config.DEPTH_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )
        self.vismerge_linear = nn.Sequential(
            nn.Linear(
                model_config.DEPTH_ENCODER.output_size + model_config.RGB_ENCODER.output_size + model_config.VISUAL_DIM.directional,
                model_config.VISUAL_DIM.vis_hidden,
            ),
            nn.ReLU(True),
        )

        self.action_state_project = nn.Sequential(
            nn.Linear(model_config.VISUAL_DIM.vis_hidden+model_config.VISUAL_DIM.directional,
            model_config.VISUAL_DIM.vis_hidden),
            nn.Tanh())
        self.action_LayerNorm = BertLayerNorm(
            model_config.VISUAL_DIM.vis_hidden, eps=layer_norm_eps)

        self.drop_env = nn.Dropout(p=0.4)
    
    
    def forward(self, 
                observations=None, 
                lang_masks=None, 
                lang_feats=None, 
                lang_token_type_ids=None,
                headings=None,
                cand_rgb=None, cand_depth=None,
                cand_direction=None,
                cand_mask=None, masks=None,
                lang_feat_type=None):
        
        headings = [2*np.pi - k for k in headings]     
        prev_actions = angle_feature_with_ele(headings, device=self.device)

        cand_rgb_feats_pool = self.space_pool(cand_rgb)
        cand_rgb_feats_pool = self.drop_env(cand_rgb_feats_pool)
        cand_depth_feats_pool = self.space_pool(cand_depth)

        rgb_in = self.rgb_linear(cand_rgb_feats_pool)
        depth_in = self.depth_linear(cand_depth_feats_pool)

        vis_in = self.vismerge_linear(
            torch.cat((rgb_in, depth_in, cand_direction), dim=2),
        )

        ''' vln-bert processing ------------------------------------- '''
        state_action_embed = torch.cat(    
            (lang_feats[:,0,:], prev_actions), dim=1)
        state_with_action = self.action_state_project(state_action_embed)       
        state_with_action = self.action_LayerNorm(state_with_action)

        self.vln_bert_lang.config.directions = cand_rgb.size(1)      

        state_feats = torch.cat((
            state_with_action.unsqueeze(1), lang_feats[:,1:,:]), dim=1)    

        bert_candidate_mask = (cand_mask == 0)     
        attention_mask = torch.cat((
            lang_masks, bert_candidate_mask), dim=-1)    

        h_t, lang_state_scores, attened_lang = self.vln_bert_lang('visual',
            state_feats,       
            attention_mask=attention_mask,     
            lang_mask=lang_masks, vis_mask=bert_candidate_mask,    
            img_feats=vis_in)     

        if lang_feat_type == 'h_t':
            my_lang_feat = h_t
        elif lang_feat_type == 'attened_lang':
            my_lang_feat = attened_lang
        elif lang_feat_type == 'lang_state_scores':
            my_lang_feat = lang_state_scores

        return my_lang_feat, h_t     
        

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias