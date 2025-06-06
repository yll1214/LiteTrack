import torch
import torch.nn.functional as F
from torch import nn

from .mae_vit import mae_vit_base_patch16, mae_vit_large_patch16

from .bert_backbone import BertModel
import numpy as np
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# from .linear import StarMLP

class ModalityUnifiedFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        """ Initializes the model."""
        super().__init__()
        self.cfg = cfg
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fusion_layer = cfg.MODEL.BACKBONE.FUSION_LAYER
        self.cont_loss_layer = cfg.MODEL.BACKBONE.CONT_LOSS_LAYER
        self.txt_token_mode = cfg.MODEL.BACKBONE.TXT_TOKEN_MODE
        self.epoch_max = cfg.TRAIN.EPOCH
        if 'base' in cfg.MODEL.BACKBONE.PRETRAINED_PATH:
            self.vit = mae_vit_base_patch16(img_size=(cfg.DATA.TEMPLATE.SIZE, cfg.DATA.SEARCH.SIZE),
                                            learnable_pos=cfg.MODEL.LEARNABLE_POSITION,
                                            drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH_RATE)
            self.vit.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_PATH, map_location='cpu')['model'],
                                     strict=False)

            # Text feature encoder (BERT)
            self.bert = BertModel.from_pretrained(cfg.MODEL.BACKBONE.LANGUAGE.TYPE)
            self.bert.encoder.layer = self.bert.encoder.layer[:min(self.fusion_layer)]

        elif 'large' in cfg.MODEL.BACKBONE.PRETRAINED_PATH:
            self.vit = mae_vit_large_patch16(img_size=(cfg.DATA.TEMPLATE.SIZE, cfg.DATA.SEARCH.SIZE),
                                             learnable_pos=cfg.MODEL.LEARNABLE_POSITION,
                                             drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH_RATE)
            self.vit.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_PATH, map_location='cpu')['model'],
                                     strict=False)

            # Text feature encoder (BERT)
            self.bert = BertModel.from_pretrained(cfg.MODEL.BACKBONE.LANGUAGE.TYPE)
            self.bert.encoder.layer = self.bert.encoder.layer[:min(self.fusion_layer)]

        for v in self.bert.pooler.parameters():
            v.requires_grad_(False)

        for name, param in self.bert.named_parameters(): 
            if any([k in name for k in [
                'embeddings', 
                'encoder.layer.0'
            ]]):
                param.requires_grad = False
        class EfficientViTFreezer:
            def __init__(self, vit_model, tracking_task='single_object'):
                """
                Args:
                    vit_model: 预训练的ViT模型
                    tracking_task: 'single_object' 或 'multi_object'
                """
                self.model = vit_model
                self.task = tracking_task
                self._init_freeze_config()
                self._apply_initial_freeze()

            def _init_freeze_config(self):
                self.config = {

                    'freeze_embeddings': True,  # patch_embed + pos_embed
                    'freeze_cls_token': False,

                    'frozen_blocks': [0, 1, 2, 3],
                    'unfreeze_norm': True,
                    'unfreeze_last_attn': True,

                    'dynamic_unfreeze': {
                        'enable': True,
                        'metric': 'loss',
                        'patience': 3,
                        'delta': 0.01,
                        'candidate_layers': [4, 5, 6]
                    },

                    'task_specific': {
                        'single_object': {
                            'unfreeze_last_ffn': True
                        },
                        'multi_object': {
                            'unfreeze_last_ffn': False
                        }
                    }
                }

                self.best_metric = float('inf')
                self.no_improve_epochs = 0
            def _apply_initial_freeze(self):
                for name, param in self.model.named_parameters():

                    if ('patch_embed' in name or 'pos_embed' in name) and self.config['freeze_embeddings']:
                        param.requires_grad = False

                    elif 'cls_token' in name and self.config['freeze_cls_token']:
                        param.requires_grad = False

                    elif any(f'blocks.{i}.' in name for i in self.config['frozen_blocks']):
                        if not (self.config['unfreeze_norm'] and 'norm' in name):
                            param.requires_grad = False
                    elif self.config['unfreeze_last_attn'] and any(
                            f'blocks.{i}.' in name for i in [9, 10, 11]) and 'attn' in name:
                        param.requires_grad = True
                    elif self.config['task_specific'][self.task]['unfreeze_last_ffn']:
                        if any(f'blocks.{i}.' in name for i in [9, 10, 11]) and 'mlp' in name:
                            param.requires_grad = True
        freezer = EfficientViTFreezer(self.vit, tracking_task='single_object')
        
        
    def cat_mask(self, text, flag):
        x_mask = torch.ones([flag.shape[0], self.vit.num_patches_x]).to(flag.device)
        z_mask = torch.ones([flag.shape[0], self.vit.num_patches_z]).to(flag.device) * (flag != 1)  # =1 mask
        c_mask = torch.ones([flag.shape[0], 1]).to(flag.device) * (flag != 1)  # =1 mask
        t_mask = text.mask * (flag != 0)  # =0 mask
        mask = ~torch.cat([c_mask, z_mask, x_mask, t_mask], dim=1).bool()
        visual_mask = ~torch.cat([c_mask, z_mask, x_mask], dim=1).bool()
        return mask, visual_mask

    def forward(self, template, search, text, flag):  # one more token
        img_feat = self.vit.patchify(template, search)
        txt_feat, bert_mask = self.bert.embedding(text.tensors, token_type_ids=None, attention_mask=text.mask)
        mask, visual_mask = self.cat_mask(text, flag)
        logits_list = []
        for i in range(len(self.vit.blocks)):
            if i in self.fusion_layer:
                img_feat, txt_feat = self.vit.forward_joint(img_feat, txt_feat, mask, i, flag=flag)
            else:
                from typing import Optional, Callable
                # StarMLP 定义
                class StarMLP(nn.Module):
                    def __init__(
                            self,
                            input_dim: int,
                            output_dim: int,
                            width_factor: int,
                            intermediate_dim: Optional[int] = None,
                            activation: Optional[Callable] = nn.ReLU6(),
                    ):
                        super().__init__()
                        self.f1 = nn.Linear(input_dim, width_factor * input_dim)
                        self.f2 = nn.Linear(input_dim, width_factor * input_dim)
                        self.act = activation  # 传入的激活函数
                        self.g = nn.Linear(width_factor * input_dim, output_dim)

                    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                        hidden_states = hidden_states.to(next(self.parameters()).device)
                        x1, x2 = self.f1(hidden_states), self.f2(hidden_states)
                        x1 = torch.clamp(x1, min=-1e3, max=1e3)
                        x2 = torch.clamp(x2, min=-1e3, max=1e3)
                        if self.act:
                            x = self.act(x1) * x2
                        else:
                            x = x1 * x2
                        x = x1 * x2
                        x = self.g(x)
                        return x

                star_mlp = StarMLP(input_dim=img_feat.shape[-1], output_dim=768, width_factor=1)
                star_mlp = star_mlp.to(img_feat.device)
                img_feat = star_mlp(img_feat)
                img_feat = self.vit.blocks[i](img_feat, visual_mask, flag=flag)
                star_mlp = StarMLP(input_dim=txt_feat.shape[-1], output_dim=768, width_factor=1)
                star_mlp = star_mlp.to(txt_feat.device)
                txt_feat = star_mlp(txt_feat)

                
                txt_feat = self.bert.encoder.layer[i](txt_feat, bert_mask)
            if i in self.cont_loss_layer:
                logits = self.contractive_learning(img_feat, txt_feat, text, flag)
                logits_list.append(logits)
        vis_token, z, x = img_feat.split([1, self.vit.num_patches_z, self.vit.num_patches_x], dim=1)
        b, s, c = x.shape
        out_dict = {
            "search": x,
            "template": z,
            "text": txt_feat,
            "vis_token": vis_token,
            "txt_token": self.generate_txt_token(txt_feat, text),
            "flag": flag.reshape(-1),
            "logits": torch.stack(logits_list, dim=1).reshape(b, -1, int(s ** 0.5), int(s ** 0.5))
        }
        return out_dict

    def generate_txt_token(self, txt_feat, text):
        if self.txt_token_mode == 'mean':
            return (txt_feat * text.mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / text.mask.unsqueeze(-1).sum(dim=1,
                                                                                                               keepdim=True)
        elif self.txt_token_mode == 'cls':
            return txt_feat[:, :1]

    def contractive_learning(self, img_feat, txt_feat, text, flag):
        vis_token, z, x = img_feat.split([1, self.vit.num_patches_z, self.vit.num_patches_x], dim=1)
        txt_token = self.generate_txt_token(txt_feat, text)
        vis_logits = self.logit_scale.exp() * (
                    F.normalize(x, dim=-1) @ F.normalize(vis_token, dim=-1).transpose(-2, -1))
        txt_logits = self.logit_scale.exp() * (
                    F.normalize(x, dim=-1) @ F.normalize(txt_token, dim=-1).transpose(-2, -1))
        logits_group = torch.stack([vis_logits, txt_logits, (vis_logits + txt_logits) / 2], dim=1)
        bid = torch.arange(flag.shape[0])
        logits = logits_group[bid, flag.reshape(-1)]
        return logits


def modality_unified_feature_extractor(cfg):
    model = ModalityUnifiedFeatureExtractor(cfg)
    return model
