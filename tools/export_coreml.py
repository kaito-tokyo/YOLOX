#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
from torch import nn

import coremltools as ct

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    final_output = []
    empty_tensor = torch.zeros((0, 7), device=prediction.device, dtype=prediction.dtype)
    for det in output:
        if det is None:
            final_output.append(empty_tensor)
        else:
            final_output.append(det)

    return final_output

class YOLOX_with_postprocess(nn.Module):
    def __init__(self, model, num_classes, conf_threshold, nms_threshold, test_size, class_agnostic=False):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.test_size = test_size # 正規化のために画像サイズを保持
        self.class_agnostic = class_agnostic

    def forward(self, x):
        preds = self.model(x)
        processed_preds = postprocess(
            preds, self.num_classes, self.conf_threshold, self.nms_threshold, self.class_agnostic
        )
        
        detections = processed_preds[0]

        # 検出がない場合の処理
        if detections is None or detections.shape[0] == 0:
            # VNCoreMLModelの仕様に合わせた空のテンソルを返す
            # coordinates: [N, 4], confidence: [N, num_classes]
            return torch.zeros((0, 4)), torch.zeros((0, self.num_classes))

        # 検出結果を VNCoreMLModel の仕様に変換
        num_dets = detections.shape[0]

        # 1. 'coordinates' テンソルの作成
        boxes = detections[:, :4]
        # (x1, y1, x2, y2) -> (center_x, center_y, width, height) & 正規化
        img_h, img_w = self.test_size
        x1 = boxes[:, 0] / img_w
        y1 = boxes[:, 1] / img_h
        x2 = boxes[:, 2] / img_w
        y2 = boxes[:, 3] / img_h
        
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        
        coordinates = torch.stack((cx, cy, w, h), dim=1)

        # 2. 'confidence' テンソルの作成
        obj_conf = detections[:, 4]
        class_conf = detections[:, 5]
        class_indices = detections[:, 6].long()
        
        scores = obj_conf * class_conf
        
        confidence = torch.zeros(num_dets, self.num_classes)
        # 各検出結果の正しいクラスインデックスにスコアを配置
        confidence[torch.arange(num_dets), class_indices] = scores
        
        return coordinates, confidence

def make_parser():
    parser = argparse.ArgumentParser("YOLOX CoreML deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.mlpackage", help="output name of models"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument("--num_classes", type=int, default=80, help="number of classes")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    exp.num_classes = args.num_classes

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, exp.exp_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = True

    conf_thre = 0.3
    nms_thre = 0.3

    wrapped_model = YOLOX_with_postprocess(
        model, exp.num_classes, conf_thre, nms_thre, exp.test_size
    )
    wrapped_model.eval()

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])

    traced_model = torch.jit.trace(wrapped_model, dummy_input)

    image_input = ct.ImageType(
        shape=dummy_input.shape,
        color_layout=ct.colorlayout.BGR
    )

    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        convert_to="mlprogram",
    )
    
    mlmodel.save(args.output_name)

    logger.info("generated CoreML model named {}".format(args.output_name))


if __name__ == "__main__":
    main()
