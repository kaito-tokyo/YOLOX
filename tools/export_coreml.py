#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
from torch import nn
import torchvision

import coremltools as ct

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


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

class YOLOX_with_postprocess(nn.Module):
    def __init__(self, model, conf_thre=0.25, nms_thre=0.45, class_agnostic=False):
        super().__init__()
        self.model = model

        self.conf_thre = conf_thre
        self.nms_thre = nms_thre

        self.class_agnostic = class_agnostic

    def forward(self, x):
        prediction = self.model(x).squeeze()
        box_corner = prediction.new(prediction.shape)
        box_corner[:, 0] = prediction[:, 0] - prediction[:, 2] / 2
        box_corner[:, 1] = prediction[:, 1] - prediction[:, 3] / 2
        box_corner[:, 2] = prediction[:, 0] + prediction[:, 2] / 2
        box_corner[:, 3] = prediction[:, 1] + prediction[:, 3] / 2
        prediction[:, :4] = box_corner[:, :4]

        class_conf, class_pred = torch.max(prediction[:, 5:], 1)

        conf_mask = prediction[:, 4] * class_conf >= self.conf_thre

        (rows,) = conf_mask.nonzero(as_tuple=True)

        confident_candidates = prediction[rows]
        confident_class_conf = class_conf[rows]
        confident_class_pred = class_pred[rows]
        confident_conf = confident_candidates[:, 4] * confident_class_conf

        if self.class_agnostic:
            nms_out_index = torchvision.ops.nms(
                confident_candidates[:, :4],
                confident_conf,
                self.nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                confident_candidates[:, :4],
                confident_conf,
                confident_class_pred,
                self.nms_thre,
            )

        final_detections = confident_candidates[nms_out_index]
        final_class_pred = confident_class_pred[nms_out_index]
        final_conf = confident_conf[nms_out_index]

        return final_detections[:, :4], final_conf, final_class_pred


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

    wrapped_model = YOLOX_with_postprocess(
        model,
        class_agnostic=True,
    )
    wrapped_model.eval()

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])

    traced_model = torch.jit.trace(wrapped_model, dummy_input)
    
    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, exp.test_size[0], exp.test_size[1]),
        color_layout=ct.colorlayout.BGR
    )

    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        outputs=[
            ct.TensorType(name="coordinates"),
            ct.TensorType(name="confidence"),
            ct.TensorType(name="labels"),
        ],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
    )
    
    mlmodel.save(args.output_name)

    logger.info("generated CoreML model named {}".format(args.output_name))


if __name__ == "__main__":
    main()
