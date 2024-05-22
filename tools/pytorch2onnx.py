# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch
from torch import nn
from collections import OrderedDict
from mmcv import Config
from mmcv.runner import wrap_fp16_model
from mmdet3d.models import build_detector

import onnx
from onnxsim import simplify
import numpy as np
import onnxruntime as ort


def parse_args():
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint file')
    # parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--no_simplify', action='store_false')
    parser.add_argument('--shape', nargs=2, type=int, default=[320, 800])
    parser.add_argument('--out_name', default='fcos3d.onnx', type=str, help="Name for the onnx output")
    parser.add_argument('--batch_size', default=1, type=int, help="batch size for exported onnx")
    parser.add_argument('--opset', default=13, type=int, help="Opset for exported onnx")
    parser.add_argument('--soft_weights_loading', action='store_true', default=False,
                        help="Loading weights in a non-strict manner.\ni.e., does not assert if there's a mismatch in the keys of state_dict and model")
    parser.add_argument('--verify', action='store_true', default=False, help="Verify ONNX output vs. Pytorch output")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    output_onnx_name = args.out_name

    # Load Config  
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # Load ckpt
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if args.soft_weights_loading:
            load_pretrained_weights_soft(model, ckpt)
        else:
            if 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                model.load_state_dict(ckpt)
    else:
        print("Using random weights")
        model.apply(init_weights)
        output_onnx_name = output_onnx_name.replace(".onnx", "_random.onnx")
    # if repvgg style -> deploy
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()

    model.eval()
    imgs = torch.randn(args.batch_size, 3, args.shape[0], args.shape[1], dtype=torch.float32).to(device)
    # dummy forward pass
    model.forward = model.forward_dummy
    pytorch_output = model(imgs)
    torch.onnx.export(model,
                      imgs,
                      output_onnx_name,
                      input_names=['test_input'],
                      output_names=['output'],
                      training=torch.onnx.TrainingMode.PRESERVE,
                      do_constant_folding=False,
                    #   verbose=False,
                      opset_version=args.opset)

    if args.no_simplify:
        model_onnx = onnx.load(output_onnx_name)
        model_simp, check = simplify(model_onnx)
        onnx.save(model_simp, output_onnx_name)
        print('Simplified model saved at: ', output_onnx_name)
    else:
        print('Model saved at: ', output_onnx_name)

    onnx.checker.check_model(output_onnx_name, full_check=True)

    # if args.verify:
    #     model_ort = ort.InferenceSession(output_onnx_name)
    #     input_names=['test_input']
    #     output_names=['output']
    #     output_onnx = model_ort.run([output_names], {input_names: imgs})
    #     np.testing.assert_allclose(pytorch_output, output_onnx[0], rtol=1e-03, atol=1e-05)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        # Initialize weights using Xavier (Glorot) uniform initialization
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def load_pretrained_weights_soft(model, checkpoint):

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for mk, mv in model_dict.items():
        # Aligning only backbone layers from state_dict into the model
        if mk.startswith('backbone.'):
            k = mk[9:]  # discard backbone.

            if k in state_dict and state_dict[k].size() == mv.size():
                new_state_dict[mk] = state_dict[k]
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        print(
            'WARNING:: The pretrained weights cannot be loaded, '
            'please check the key names manually '
        )
    else:
        print('Successfully loaded pretrained weights')
        if len(discarded_layers) > 0:
            print(
                'WARNING:: ** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(epilog='Example: CUDA_VISIBLE_DEVICES=0 python tools/pytorch2onnx.py configs/fcn/fcn8_r18_hailo.py --checkpoint work_dirs/fcn8_r18_hailo_iterbased/epoch_1.pth --out_name my_fcn_model.onnx --shape 608 800')
    main()
