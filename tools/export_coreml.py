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
from coremltools.converters.mil.mil import Builder as mb
import coremltools.proto.Model_pb2 as ml_spec
from coremltools.models.utils import rename_feature

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
        "--nms_thre", 
        type=float, 
        default=0.3, 
        help="NMS IoU threshold"
    )
    parser.add_argument(
        "--conf_thre", 
        type=float, 
        default=0.3, 
        help="Confidence threshold for filtering"
    )
    parser.add_argument(
        "--max_boxes", 
        type=int, 
        default=100, 
        help="Maximum number of boxes to output after NMS"
    )
    parser.add_argument(
        "--class_agnostic",
        action="store_true",
        help="Enable class-agnostic NMS"
    )

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

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])

    traced_model = torch.jit.trace(model, dummy_input)
    
    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, exp.test_size[0], exp.test_size[1]),
        color_layout=ct.colorlayout.BGR
    )

    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        outputs=[
            ct.TensorType(name="prediction"),
        ],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
    )

    NUM_CLASSES = args.num_classes
    NMS_THRESHOLD = 0.3
    CONF_THRESHOLD = 0.3

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, box_size, 5 + args.num_classes))])
    def postprocess_program(prediction):
        coordinates_all = mb.slice_by_index(
            x=prediction, begin=[0, 0, 0], end=[1, box_size, 4]
        )
        obj_conf_all = mb.slice_by_index(
            x=prediction, begin=[0, 0, 4], end=[1, box_size, 5]
        )
        class_confs_all = mb.slice_by_index(
            x=prediction, begin=[0, 0, 5], end=[1, box_size, 5 + args.num_classes]
        )

        scores_all = mb.mul(x=obj_conf_all, y=class_confs_all)

        final_coordinates, final_scores, _, _ = mb.non_maximum_suppression(
            boxes=coordinates_all,
            scores=scores_all,
            iou_threshold=args.nms_thre,
            score_threshold=args.conf_thre,
            max_boxes=args.max_boxes,
            per_class_suppression=not args.class_agnostic,
            name="nms",
        )
        return final_coordinates, final_scores

    mlmodel_spec = mlmodel.get_spec()

    postprocess_model = ct.convert(
        postprocess_program,
        convert_to="mlprogram"
    )
    postprocess_spec = postprocess_model.get_spec()
    rename_feature(postprocess_spec, "nms_0", "coordinates")
    rename_feature(postprocess_spec, "nms_1", "confidence")

    pipeline_spec = ml_spec.Model()
    pipeline_spec.specificationVersion = ct.SPECIFICATION_VERSION

    pipeline_spec.description.input.extend(mlmodel_spec.description.input)
    pipeline_spec.description.output.extend(postprocess_spec.description.output)
    
    pipeline = pipeline_spec.pipeline
    pipeline.models.add().CopyFrom(mlmodel_spec)
    pipeline.models.add().CopyFrom(postprocess_spec)

    final_pipeline_model = ct.models.MLModel(pipeline_spec, weights_dir=mlmodel.weights_dir)
    final_pipeline_model.save(args.output_name)

    logger.info("generated CoreML model named {}".format(args.output_name))


if __name__ == "__main__":
    main()
