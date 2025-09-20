#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
from torch import nn
import torchvision  # torchvisionを追加

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
        "--conf", default=0.25, type=float, help="test conf threshold"
    )
    parser.add_argument(
        "--nms", default=0.45, type=float, help="test nms threshold"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

# -------------------- ここから追加/変更 --------------------

class PostProcess(nn.Module):
    """
    後処理を行い、出力を固定シェイプにパディングするPyTorchモジュール (トレース安全版)
    """
    def __init__(self, num_classes: int, conf_thre: float, nms_thre: float, max_detections: int = 100):
        super().__init__()
        self.num_classes = num_classes
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre
        self.max_detections = max_detections

    def forward(self, prediction: torch.Tensor):
        prediction = prediction[0]

        # 座標変換
        box_corner = torch.empty_like(prediction[:, :4])
        box_corner[:, 0] = prediction[:, 0] - prediction[:, 2] / 2
        box_corner[:, 1] = prediction[:, 1] - prediction[:, 3] / 2
        box_corner[:, 2] = prediction[:, 0] + prediction[:, 2] / 2
        box_corner[:, 3] = prediction[:, 1] - prediction[:, 3] / 2

        # スコア計算とフィルタリング
        class_conf, class_pred = torch.max(prediction[:, 5: 5 + self.num_classes], 1, keepdim=True)
        conf_mask = (prediction[:, 4] * class_conf.squeeze() >= self.conf_thre).squeeze()
        detections = torch.cat((box_corner, prediction[:, 4:5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
            
        # NMS
        nms_out_index = torchvision.ops.nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            self.nms_thre,
        )
        detections_after_nms = detections[nms_out_index]

        # --- ▼▼▼ if文を使わないパディング処理 ▼▼▼ ---

        # 1. if文を使わずに最大検出数で切り捨てる
        #    要素数がmax_detectionsより少なくても、スライスはエラーなく全要素を返す
        detections_after_nms = detections_after_nms[:self.max_detections]
        
        # 2. 実際の検出数を取得
        num_actual_detections = detections_after_nms.shape[0]

        # CoreMLの出力として分かりやすいように3つに分割
        boxes = detections_after_nms[:, :4]
        scores = detections_after_nms[:, 4] * detections_after_nms[:, 5]
        class_ids = detections_after_nms[:, 6]
        
        # 固定長のテンソルを作成
        padded_boxes = torch.zeros(self.max_detections, 4)
        padded_scores = torch.zeros(self.max_detections)
        padded_class_ids = torch.zeros(self.max_detections)

        # 3. if文を使わずに結果をコピー
        #    num_actual_detectionsが0の場合、スライス[:0]は空になり、何もコピーされず安全
        padded_boxes[:num_actual_detections] = boxes
        padded_scores[:num_actual_detections] = scores
        padded_class_ids[:num_actual_detections] = class_ids

        return padded_boxes, padded_scores, padded_class_ids


class YoloXWithPostProcess(nn.Module):
    """
    YOLOXモデルと後処理モジュールを結合するラッパー
    """
    def __init__(self, model: nn.Module, post_process: nn.Module):
        super().__init__()
        self.model = model
        self.post_process = post_process
    
    def forward(self, x: torch.Tensor):
        predictions = self.model(x)
        boxes, scores, class_ids = self.post_process(predictions)
        return boxes, scores, class_ids

# -------------------- ここまで追加/変更 --------------------

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
    
    # 1. モデル内部のデコード処理を有効にする
    model.head.decode_in_inference = True

    # 2. 後処理モジュールをインスタンス化
    post_process = PostProcess(
        num_classes=exp.num_classes, 
        conf_thre=args.conf, 
        nms_thre=args.nms
    )

    # 3. YOLOXモデルと後処理モジュールを結合
    combined_model = YoloXWithPostProcess(model, post_process)
    combined_model.eval()

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])

    # 4. 結合したモデルをトレース
    traced_model = torch.jit.trace(combined_model, dummy_input)
    
    image_input = ct.ImageType(
        name="image",
        shape=dummy_input.shape,
        scale=1/255.0,
        color_layout=ct.colorlayout.BGR
    )

    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        convert_to="mlprogram",
    )
    
    # --- ▼▼▼ ここからリネーム処理 (より確実な方法に修正) ▼▼▼ ---
    logger.info("Renaming model outputs...")
    spec = mlmodel.get_spec()
    
    # 重みファイルが保存されている一時ディレクトリのパスを取得
    weights_dir = mlmodel.weights_dir

    # 出力名をリネーム
    original_output_names = [o.name for o in spec.description.output]
    new_output_names = ["boxes", "scores", "class_labels"]

    if len(original_output_names) == len(new_output_names):
        for i, new_name in enumerate(new_output_names):
            original_name = original_output_names[i]
            ct.utils.rename_feature(spec, original_name, new_name)
            logger.info(f"Renamed output '{original_name}' to '{new_name}'")
    else:
        logger.warning("Could not rename outputs due to a mismatch in the number of outputs.")

    # 変更済みのspecと重みファイルのパスから、新しいMLModelオブジェクトを明示的に生成
    mlmodel_renamed = ct.models.MLModel(spec, weights_dir=weights_dir)
    
    # 新しいオブジェクトを保存
    mlmodel_renamed.save(args.output_name)

    logger.info("generated CoreML model named {} with post-processing.".format(args.output_name))

if __name__ == "__main__":
    main()
