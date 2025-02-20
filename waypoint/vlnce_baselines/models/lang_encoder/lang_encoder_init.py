from vlnce_baselines.models.lang_encoder.lang_encoders import Lang_by_VLN_BERT
import torch
import torch.nn as nn


class Lang_Encoder_by_VLN_BERT(nn.Module):
    def __init__(
        self,
        model_config,
        device,
        trainable=False,
        checkpoint="NONE",
    ):
        super().__init__()
        self.model = Lang_by_VLN_BERT(model_config)

        self.device=device


        for param in self.model.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            encoder_weights = torch.load(checkpoint, map_location=torch.device('cpu'))

            ckpt_dict = {}

            for k, v in encoder_weights["state_dict"].items():
                if 'module' in k:      
                    name_idx = 2
                else:
                    name_idx = 1
                split_layer_name = k.split(".")
                if split_layer_name[name_idx] == "depth_encoder" or split_layer_name[name_idx] == "rgb_encoder":
                    continue
                
                if split_layer_name[name_idx] == 'vln_bert':
                    split_layer_name[name_idx] = 'vln_bert_lang'
                layer_name = ".".join(split_layer_name[1:])
                ckpt_dict[layer_name] = v
            
            del encoder_weights

            if 'module' in list(ckpt_dict.keys())[0]:
                self.model = nn.DataParallel(self.model.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                self.model.load_state_dict(ckpt_dict, strict=True)
                self.model = self.model.module
                
            else:
                self.model.load_state_dict(ckpt_dict, strict=True)
            
            
            

            
    def forward(self, observations=None, 
                lang_masks=None, 
                lang_feats=None, 
                lang_token_type_ids=None,
                headings=None,
                cand_rgb=None, cand_depth=None,
                cand_direction=None,
                cand_mask=None, masks=None,
                lang_feat_type=None):
        
        return self.model(
                    observations=observations, 
                    lang_masks=lang_masks, 
                    lang_feats=lang_feats, 
                    lang_token_type_ids=lang_token_type_ids,
                    headings=headings,
                    cand_rgb=cand_rgb, cand_depth=cand_depth,
                    cand_direction=cand_direction,
                    cand_mask=cand_mask, masks=masks,
                    lang_feat_type=lang_feat_type)
        


    