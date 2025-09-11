#!/usr/bin/env python
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved

import setuptools
import sys

TORCH_AVAILABLE = True
try:
    import torch
    from torch.utils import cpp_extension
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] Unable to import torch, pre-compiling ops will be disabled.")


def get_package_dir():
    pkg_dir = {
        "yolox.tools": "tools",
        "yolox.exp.default": "exps/default",
    }
    return pkg_dir


def get_ext_modules():
    ext_module = []
    if sys.platform != "win32":  # pre-compile ops on linux
        assert TORCH_AVAILABLE, "torch is required for pre-compiling ops, please install it first."
        # if any other op is added, please also add it here
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "yolox.layers.jit_ops",
            os.path.join("yolox", "layers", "jit_ops.py"),
        )
        jit_ops = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(jit_ops)
        ext_module.append(jit_ops.FastCOCOEvalOp().build_op())
    return ext_module


def get_cmd_class():
    cmdclass = {}
    if TORCH_AVAILABLE:
        cmdclass["build_ext"] = cpp_extension.BuildExtension
    return cmdclass


setuptools.setup(
    package_dir=get_package_dir(),
    packages=setuptools.find_packages(exclude=("tests", "tools")) + list(get_package_dir().keys()),
    ext_modules=get_ext_modules(),
    cmdclass=get_cmd_class(),
)
