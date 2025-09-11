# Development Guideline

## How to start development
1. Run `python3.11 -m venv .venv` to create a virtual env,
2. Run `.venv/bin/pip3 install --upgrade pip` to update pip.
3. Run `.venv/bin/pip3 install -r requirements.txt` to install dependencies.
4. Run `.venv/bin/pip3 install -v -e .` to install YOLOX.
5. Download pre-trained weights and place them under the project root. Their URLs are available on README.md.

## How to train custom data
Custom data training requires NVidia GPU.
Before training, you must have started development. You can help the procedure.
Refer docs/train_custom_data.md for further details.
An example training which uses coco128 is available on train_coco128.ipynb.
