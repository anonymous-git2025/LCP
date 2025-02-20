import torch
import torch.nn as nn
import numpy as np
from .utils import get_attention_mask

from .transformer.waypoint_bert import WaypointBert
from pytorch_transformers import BertConfig

class BinaryDistPredictor_TRM_lang(nn.Module):
    def __init__(self, hidden_dim=768, n_classes=12, device=None, lang=None):
        super(BinaryDistPredictor_TRM_lang, self).__init__()

        self.device = device

        self.num_angles = 120
        self.num_imgs = 12
        self.n_classes = 12  
        self.TRM_LAYER = 2
        self.TRM_NEIGHBOR = 1
        self.HEATMAP_OFFSET = 5

        self.lang = lang

        self.visual_fc_rgb = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod([2048,7,7]), hidden_dim),
            nn.ReLU(True),
        )
        self.visual_fc_depth = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod([128,4,4]), hidden_dim),
            nn.ReLU(True),
        )
        self.visual_merge = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(True),
        )


        if self.lang == 'h_t' or self.lang == 'attened_lang':
            self.lang_fc = nn.Sequential(
                nn.Linear(768, hidden_dim),
                nn.ReLU(True),
            )
        elif self.lang == 'lang_state_scores':
            self.lang_fc = nn.Sequential(
                nn.Linear(79, hidden_dim),
                nn.ReLU(True),
            )

        if self.lang:
            self.lang_merge = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(True),
            )

        config = BertConfig()
        config.model_type = 'visual'
        config.finetuning_task = 'waypoint_predictor'
        config.hidden_dropout_prob = 0.3
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = self.TRM_LAYER
        self.waypoint_TRM = WaypointBert(config=config)

        self.mask = get_attention_mask(
            num_imgs=self.num_imgs,
            neighbor=self.TRM_NEIGHBOR).to(self.device)

        self.vis_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,
                int(n_classes*(self.num_angles/self.num_imgs))),
        )

    def forward(self, rgb_feats, depth_feats, lang_feats):
        bsi = rgb_feats.size(0) // self.num_imgs

        lang_feats = lang_feats.reshape(bsi, 1, -1)     

        rgb_x = self.visual_fc_rgb(rgb_feats).reshape(
            bsi, self.num_imgs, -1)
        depth_x = self.visual_fc_depth(depth_feats).reshape(
            bsi, self.num_imgs, -1)
        
        lang_x = self.lang_fc(lang_feats).repeat(1, self.num_imgs, 1)

        vis_x = self.visual_merge(
            torch.cat((rgb_x, depth_x), dim=-1)
        )
        
        lang_vis_x = self.lang_merge(
            torch.cat((vis_x, lang_x), dim=-1)
        )

        attention_mask = self.mask.repeat(bsi,1,1,1)
        vis_rel_x = self.waypoint_TRM(
            lang_vis_x, attention_mask=attention_mask
        )

        vis_logits = self.vis_classifier(vis_rel_x)
        vis_logits = vis_logits.reshape(
            bsi, self.num_angles, self.n_classes)

     
        vis_logits = torch.cat(
            (vis_logits[:,self.HEATMAP_OFFSET:,:], vis_logits[:,:self.HEATMAP_OFFSET,:]),
            dim=1)

        return vis_logits


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
