from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target, grounding_resize
# for debug
from copy import deepcopy
import cv2
import os
from lib.utils.merge import merge_template_search
from lib.models.LiteTrack.LiteTrack import build_model
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box, box_xywh_to_xyxy, box_cxcywh_to_xywh, box_cxcywh_to_xyxy
import numpy as np
import matplotlib.pyplot as plt
from lib.test.utils.hann import hann2d

#from pytorch_pretrained_bert import BertTokenizer
from transformers import BertModel
from transformers import BertTokenizer
from lib.utils.misc import NestedTensor

class LiteTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(LiteTrack, self).__init__(params)
        network = build_model(params.cfg)
        #network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)['net'],
                                strict=False)
        self.map_size = params.search_size // 16
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = self.params.debug
        self.frame_id = 0
        # if self.debug:
        #     self.save_dir = "/ssd/myc/VL_project/MUTrack/debug"
        #     if not os.path.exists(self.save_dir):
        #         os.makedirs(self.save_dir)
        self.update_interval = self.cfg.TEST.UPDATE_INTERVAL
        self.feat_size = self.params.search_size // 16
        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.MODEL.BACKBONE.LANGUAGE.VOCAB_PATH, do_lower_case=True)
        self.threshold = self.params.cfg.TEST.THRESHOLD
        self.has_cont = self.params.cfg.TRAIN.CONT_WEIGHT > 0
        self.max_score = 0

        self.sequence_index=0
        self.sequence_id = 1
        self.sequence_ids = list(range(1, 1001))
        self.frame_counter = 1


    def grounding(self, image, info: dict):
        bbox = torch.tensor([0., 0., 0., 0.]).cuda()
        h, w = image.shape[:2]
        im_crop_padded, _, _, _, _ = grounding_resize(image, self.params.grounding_size, bbox, None)
        ground = self.preprocessor.process(im_crop_padded).cuda()
        template = torch.zeros([1, 3, self.params.template_size, self.params.template_size]).cuda()
        template_mask = torch.zeros([1, (self.params.template_size//16)**2]).bool().cuda()
        context_mask = torch.zeros([1, (self.params.search_size//16)**2]).bool().cuda()
        text, mask = self.extract_token_from_nlp(info['language'], self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN)
        self.text = NestedTensor(text, mask)
        flag = torch.tensor([[1]]).cuda()
        with torch.no_grad():
            out_dict = self.network.forward(template, ground, self.text, template_mask, context_mask, flag)
        out_dict['pred_boxes'] = box_cxcywh_to_xywh(out_dict['pred_boxes']*np.max(image.shape[:2]))[0, 0].cpu().tolist()
        dx, dy = min(0, (w-h)/2), min(0, (h-w)/2)
        out_dict['pred_boxes'][0] = out_dict['pred_boxes'][0] + dx
        out_dict['pred_boxes'][1] = out_dict['pred_boxes'][1] + dy
        return out_dict

    def window_prior(self):
        hanning = np.hanning(self.map_size)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        self.torch_window = hann2d(torch.tensor([self.map_size, self.map_size]).long(), centered=True).flatten()

    def initialize(self, image, info: dict):
        if self.cfg.TEST.MODE == 'NL':
            grounding_state = self.grounding(image, info)
            init_bbox = grounding_state['pred_boxes']
            self.flag = torch.tensor([[2]]).cuda()
        elif self.cfg.TEST.MODE == 'NLBBOX':
            text, mask = self.extract_token_from_nlp(info['language'], self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN)
            self.text = NestedTensor(text, mask)
            init_bbox = info['init_bbox']
            self.flag = torch.tensor([[2]]).cuda()
        else:
            text = torch.zeros([1, self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN]).long().cuda()
            mask = torch.zeros([1, self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN]).cuda()
            self.text = NestedTensor(text, mask)
            init_bbox = info['init_bbox']
            self.flag = torch.tensor([[0]]).cuda()
        self.window_prior()
        z_patch_arr, _, _, bbox = sample_target(image, init_bbox, self.params.template_factor,
                                                    output_sz=self.params.template_size, return_bbox=True)
        self.template_mask = self.anno2mask(bbox.reshape(1, 4), size=self.params.template_size//16)
        self.z_patch_arr = z_patch_arr
        self.template_bbox = (bbox*self.params.template_size)[0, 0].tolist()
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        # forward the context once
        y_patch_arr, _, _, y_bbox = sample_target(image, init_bbox, self.params.search_factor,
                                                    output_sz=self.params.search_size, return_bbox=True)
        self.y_patch_arr = y_patch_arr
        self.context_bbox = (y_bbox*self.params.search_size)[0, 0].tolist()
        context = self.preprocessor.process(y_patch_arr)
        context_mask = self.anno2mask(y_bbox.reshape(1, 4), self.params.search_size//16)
        self.prompt = self.network.forward_prompt_init(self.template, context, self.text, self.template_mask, context_mask, self.flag)
        # save states
        self.state = init_bbox


    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        self.sequence_ids = list(range(1, 1001))

        if not hasattr(self, 'save_dir'):
            self.save_dir = "test/map"
            os.makedirs(self.save_dir, exist_ok=True)

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            out_dict = self.network.forward_test(self.template, search, self.text, self.prompt, self.flag)

        pred_boxes = out_dict['bbox_map'].view(-1, 4).detach().cpu()
        pred_cls = out_dict['cls_score_test'].view(-1).detach().cpu()
        pred_cont = out_dict['cont_score'].softmax(-1)[:, :, 0].view(-1).detach().cpu() if self.has_cont else 1
        pred_cls_merge = pred_cls * self.window * pred_cont
        pred_box_net = pred_boxes[torch.argmax(pred_cls_merge)]
        score = (pred_cls * pred_cont)[torch.argmax(pred_cls_merge)]

        pred_box = (pred_box_net * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)


        if not hasattr(self, 'sequence_save_dir'):
            self.sequence_save_dir = os.path.join(self.save_dir, f'sequence_{self.sequence_id}')
            os.makedirs(self.sequence_save_dir, exist_ok=True)

        self._save_heatmap_overlay(image, out_dict, resize_factor)

        if self.frame_id == 1:
            self.sequence_index = self.sequence_index + 1
            #print(self.sequence_index)
            #print(self.sequence_ids)
            self.sequence_id = self.sequence_ids[self.sequence_index]
            #print(self.sequence_id )
            self.current_sequence_dir = os.path.join(self.save_dir, f'sequence_{self.sequence_id}')
            os.makedirs(self.current_sequence_dir, exist_ok=True)
            #print(f"New sequence started! Saving to: {self.current_sequence_dir}")
            self.frame_counter = 1
        else:
            self.frame_counter += 1

        return {"target_bbox": self.state}
    def _save_heatmap_overlay(self, original_image, out_dict, resize_factor):
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)
        heatmap = out_dict['cls_score_test'].squeeze().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        x, y, w, h = map(int, self.state)  # [x,y,w,h]
        img_h, img_w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = original_image.copy()
        if h > 0 and w > 0:
            overlay[y:y + h, x:x + w] = cv2.addWeighted(
                original_image[y:y + h, x:x + w], 0.5,
                heatmap_color, 0.5, 0
            )
        blue_value = heatmap_color[0, 0]
        blue_heatmap = np.full((img_h, img_w, 3), blue_value, dtype=np.uint8)
        overlay = cv2.addWeighted(original_image, 0.5, blue_heatmap, 0.5, 0)
        if h > 0 and w > 0:
            overlay[y:y + h, x:x + w] = cv2.addWeighted(
                original_image[y:y + h, x:x + w], 0.5,
                heatmap_color, 0.5, 0
            )
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        overlay_path = os.path.join(self.sequence_save_dir, f'overlay_{self.frame_id:04d}.png')
        cv2.imwrite(overlay_path, overlay)
        # original_image_path = os.path.join(self.sequence_save_dir, f'original_image_{self.frame_id:04d}.png')
        # cv2.imwrite(original_image_path, original_image)



    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)
        
    def anno2mask(self, gt_bboxes, size):
        bboxes = box_xywh_to_xyxy(gt_bboxes)*size # b, 4
        cood = torch.arange(size).unsqueeze(0).repeat(gt_bboxes.shape[0], 1)+0.5 # b, sz
        x_mask = ((cood > bboxes[:, 0:1]) & (cood < bboxes[:, 2:3])).unsqueeze(1) # b, 1, w
        y_mask = ((cood > bboxes[:, 1:2]) & (cood < bboxes[:, 3:4])).unsqueeze(2) # b, h, 1
        mask = (x_mask & y_mask)

        cx = ((bboxes[:, 0]+bboxes[:, 2])/2).long()
        cy = ((bboxes[:, 1]+bboxes[:, 3])/2).long()
        bid = torch.arange(cx.shape[0]).to(cx)
        mask[bid, cy, cx] = True
        return mask.flatten(1).cuda()
    
    def extract_token_from_nlp(self, nlp, seq_length):
        """ use tokenizer to convert nlp to tokens
        param:
            nlp:  a sentence of natural language
            seq_length: the max token length, if token length larger than seq_len then cut it,
            elif less than, append '0' token at the reef.
        return:
            token_ids and token_marks
        """
        nlp_token = self.tokenizer.tokenize(nlp)
        if len(nlp_token) > seq_length - 2:
            nlp_token = nlp_token[0:(seq_length - 2)]
        # build tokens and token_ids
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in nlp_token:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        return torch.tensor(input_ids).unsqueeze(0).cuda(), torch.tensor(input_mask).unsqueeze(0).cuda()


def get_tracker_class():
    return LiteTrack
