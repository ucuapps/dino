# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import pickle
import random

import numpy as np
import sklearn.metrics as metrics

from tqdm import tqdm

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def make_predictions_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        pth_transforms.Normalize((0.5), (0.25)),

    ])
    dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "val"), transform=transform)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_val)} val imgs.")

    # ============ building network ... ============

    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
    else:
        print(f"Unknow architecture: {args.arch}")

    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")

    embed_dim = model.fc.weight.shape[1]
    model = utils.MultiCropWrapper(
        model,
        vits.DINOHead(embed_dim, args.out_dim, args.use_bn_in_head), is_student=True
    )

    model.cuda()

    # Loading weights
    if args.arch in vits.__dict__.keys():
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    else:
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

    model.eval()

    # ============ make predictions ... ============

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    predicted_labels_all = []
    with tqdm(data_loader_val, unit="batch") as tepoch:
        for it, (images, labels) in enumerate(tepoch):
            with torch.no_grad():
                _, predicted_labels = model(images.cuda(), is_student=True)
                predicted_labels = np.array([x.item() for x in predicted_labels])
                predicted_labels = sigmoid(predicted_labels)
                predicted_labels_all.extend(predicted_labels)

    predicted_labels_all = np.array(predicted_labels_all)
    print(np.max(predicted_labels_all))

    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    return predicted_labels_all, test_labels.numpy()


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
                        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
                        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'] + torchvision_archs, help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
                        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = 1
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    y_pred, y_true = make_predictions_pipeline(args)
    with open('predictions.pcl', 'wb') as f:
        pickle.dump({
            'y_pred': y_pred.tolist(),
            'y_true': y_true.tolist(),
        }, f)

    print(sum(y_true), '/', len(y_true))

    best_f1_score = 0
    best_result = {}

    for threshold in np.arange(0.05, 0.9, 0.005):
        y_pred_th = np.where(y_pred.copy() <= threshold, 0, 1)
        top1 = sum(y_pred_th == y_true) / len(y_true)
        f1_score = metrics.f1_score(y_true, y_pred_th, average='binary')
        if f1_score > best_f1_score:
            conf_matrix = metrics.confusion_matrix(y_true, y_pred_th)
            best_result['threshold'] = threshold
            best_result['top1'] = top1
            best_result['f1_score'] = f1_score
            best_result['conf_matrix'] = conf_matrix
            best_f1_score = f1_score

    threshold, top1, f1_score, conf_matrix = \
        best_result['threshold'], best_result['top1'], best_result['f1_score'], best_result['conf_matrix']

    print(f"Classification head result (threshold={threshold}): "
          f"Top1: {top1}, 'F1-score: {f1_score}, '\n' Confusion Matrix: {conf_matrix}")

    # if utils.get_rank() == 0:
    #     if args.use_cuda:
    #         train_features = train_features.cuda()
    #         test_features = test_features.cuda()
    #         train_labels = train_labels.cuda()
    #         test_labels = test_labels.cuda()
    #
    #     print("Features are ready!\nStart the k-NN classification.")
    #     for k in args.nb_knn:
    #         top1, f1_score, conf_matrix = knn_classifier(train_features, train_labels,
    #                                                      test_features, test_labels, k, args.temperature)
    #         print(f"{k}-NN classifier result: Top1: {top1}, 'F1-score: {f1_score}, '\n' Confusion Matrix: "
    #               f"{conf_matrix}")
    # dist.barrier()
