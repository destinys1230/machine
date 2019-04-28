import cv2
import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from \
    transformer_net import TransformerNet
from vgg import Vgg16


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.contentimage, scale=args.contentscale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    if args.model.endswith(".onnx"):
        output = stylize_onnx_caffe2(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(args.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            if args.exportonnx:
                assert args.exportonnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(style_model, content_image, args.export_onnx).cpu()
            else:
                output = style_model(content_image).cpu()
    utils.save_image(args.outputimage, output[0])


def stylize_onnx_caffe2(content_image, args):
    """
    Read ONNX model and run it using Caffe2
    """
    assert not args.export_onnx

    import onnx
    import onnx_caffe2.backend

    model = onnx.load(args.model)

    prepared_backend = onnx_caffe2.backend.prepare(model, device='CUDA' if args.cuda else 'CPU')
    inp = {model.graph.input[0].name: content_image.numpy()}
    c2_out = prepared_backend.run(inp)[0]

    return torch.from_numpy(c2_out)
def main():
    class hjzstr:
        def __init__(self):
            self.conterntimage = ''
            self.contentscale = None
            self.outputimage= ''
            self.outputimage=''
            self.cuda=1
            self.exportonnx=''
    args=hjzstr()
    args.model="models/hjz.pth"
    args.cuda=1

    capture = cv2.VideoCapture(0)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    while (True):
        # 获取一帧
        ret, frame = capture.read()
        cv2.imwrite("kk.jpg",frame);
        args.contentimage = "kk.jpg"
        args.outputimage = "visimg/kk1.jpg"
        stylize(args)

        frame=cv2.imread("visimg/kk1.jpg")

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()
