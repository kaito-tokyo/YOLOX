# Development Guideline

## How to start development

1. Run `python3.12 -m venv .venv` to create a virtual env,
2. Run `.venv/bin/pip3 install --upgrade pip` to update pip.
3. Run `.venv/bin/pip3 install -r requirements.txt` to install dependencies.
4. Run `.venv/bin/pip3 install -v -e .` to install YOLOX.
5. Download pre-trained weights and place them under the project root. Their URLs are available on README.md.

## How to train custom data
Custom data training requires NVidia GPU.
Before training, you must have started development. You can help the procedure.
Refer docs/train_custom_data.md for further details.
An example training which uses coco128 is available on train_coco128.ipynb.

# How to convert a model (.pth) for ONNX Runtime (.onnx)

Do not use onnxsim or onnx-simplifier.
Note that *.onnx and *.pth are git-ignored.

1. Ask the user to input the number of class and the model name.
2. Convert the .pth model into .onnx according to demo/ONNXRuntime/README.md.
   Run export_onnx.py using `.venv/bin/python3 export_onnx.py`.
   The --no-onnxsim option was removed and do not specify this.

# How to convert a model (,pth) for CoreML (.mlpackage)

Use coremltools.