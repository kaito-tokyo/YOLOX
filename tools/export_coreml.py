#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import numpy as np
import torch
from torch import nn

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
import coremltools.proto.Model_pb2 as ml_spec

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
    model.head.decode_in_inference = False

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])

    traced_model = torch.jit.trace(model, dummy_input)
    
    image_input = ct.ImageType(
        name="image",
        shape=dummy_input.shape,
        color_layout=ct.colorlayout.BGR
    )

    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        convert_to="mlprogram",
        outputs=[ct.TensorType(name="yolox_output")],
    )

    mlmodel_spec = mlmodel.get_spec()

    # --- START: Pre-calculate grids in Python/NumPy (Most robust approach) ---
    num_classes = args.num_classes
    strides_config = [
        {"stride": 8, "grid_points": 6400},
        {"stride": 16, "grid_points": 1600},
        {"stride": 32, "grid_points": 400},
    ]
    test_size = exp.test_size
    h_in, w_in = test_size
    
    precalculated_grids = {}
    for config in strides_config:
        stride = config["stride"]
        hsize, wsize = h_in // stride, w_in // stride
        yv, xv = np.meshgrid(np.arange(hsize), np.arange(wsize), indexing='ij')
        grid = np.stack((xv, yv), 2).reshape(-1, 2)
        precalculated_grids[stride] = grid.astype(np.float32)
    # --- END: Pre-calculation ---


    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 8400, 5 + num_classes))])
    def postprocess_program(yolox_output):
        yolox_output = mb.squeeze(x=yolox_output, axes=[0], name="squeeze_output")

        decoded_xywh_parts = []
        obj_conf_parts = []
        cls_confs_parts = []
        
        current_row = 0
        for config in strides_config:
            stride = config["stride"]
            grid_points = config["grid_points"]
            
            output_part = mb.slice_by_index(
                x=yolox_output,
                begin=[current_row, 0],
                end=[current_row + grid_points, 0],
                begin_mask=[False, True],
                end_mask=[False, True],
                name=f"slice_stride_{stride}"
            )
            
            # Use the pre-calculated NumPy grid as a constant. This avoids dynamic generation errors.
            grid_flat = mb.const(val=precalculated_grids[stride])

            # Use a scalar const for the stride and rely on broadcasting.
            stride_const = mb.const(val=np.float32(stride), name=f"stride_const_{stride}")

            xy_encoded = mb.slice_by_index(x=output_part, begin=[0,0], end=[0,2], begin_mask=[True, False], end_mask=[True, False])
            wh_encoded = mb.slice_by_index(x=output_part, begin=[0,2], end=[0,4], begin_mask=[True, False], end_mask=[True, False])
            obj_conf = mb.slice_by_index(x=output_part, begin=[0,4], end=[0,5], begin_mask=[True, False], end_mask=[True, False])
            cls_confs = mb.slice_by_index(x=output_part, begin=[0,5], end=[0,0], begin_mask=[True, False], end_mask=[True, True])
            
            xy_decoded = mb.mul(x=mb.add(x=xy_encoded, y=grid_flat), y=stride_const)
            wh_decoded = mb.mul(x=mb.exp(x=wh_encoded), y=stride_const)
            xywh_decoded = mb.concat(values=[xy_decoded, wh_decoded], axis=1, name=f"xywh_{stride}")

            decoded_xywh_parts.append(xywh_decoded)
            obj_conf_parts.append(obj_conf)
            cls_confs_parts.append(cls_confs)
            
            current_row += grid_points

        final_xywh = mb.concat(values=decoded_xywh_parts, axis=0, name="xywh")
        final_obj_conf = mb.concat(values=obj_conf_parts, axis=0, name="obj_conf")
        final_cls_confs = mb.concat(values=cls_confs_parts, axis=0, name="cls_confs")

        return final_xywh, final_obj_conf, final_cls_confs

    postprocess_model = ct.convert(
        postprocess_program,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL # CPU_ONLYより汎用的なALLを推奨
    )
    postprocess_spec = postprocess_model.get_spec()

    pipeline_spec = ml_spec.Model()
    pipeline_spec.specificationVersion = ct.SPECIFICATION_VERSION

    pipeline_spec.description.input.extend(mlmodel_spec.description.input)
    pipeline_spec.description.output.extend(postprocess_spec.description.output)
    
    output_names = ["xywh", "obj_conf", "cls_confs"]
    for i, out in enumerate(pipeline_spec.description.output):
        out.name = output_names[i]

    pipeline = pipeline_spec.pipeline
    pipeline.models.add().CopyFrom(mlmodel_spec)
    pipeline.models.add().CopyFrom(postprocess_spec)

    final_pipeline_model = ct.models.MLModel(pipeline_spec, weights_dir=mlmodel.weights_dir)
    final_pipeline_model.save(args.output_name)

    logger.info("generated CoreML model named {}".format(args.output_name))


if __name__ == "__main__":
    main()