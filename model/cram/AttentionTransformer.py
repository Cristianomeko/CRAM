# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt, image as mpimg
from torch import nn
import torch.nn.functional as F

from aspp import ASPP_CBAM
from seblock import *


def visualize1(x):
    x = x.view(2, 160, 15, 15).detach().cpu().numpy()
    # x = np.transpose(x, (0, 1, 3, 2))
    n = x.shape[0]
    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=(4 * n, 4))
    for k in range(n):
        axs[k].imshow(x[k], cmap=None)
        axs[k].axis('off')

    plt.show()


def visualize(x):
    x = x.detach().cpu().numpy()
    x = np.transpose(x, (0, 2, 3, 1))
    # print(x.shape)
    n = x.shape[0]
    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=(4 * n, 4))
    for k in range(n):
        axs[k].imshow(x[k], cmap=None)
        axs[k].axis('off')

    plt.show()


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(5, 5), cmap='Reds'):
    """显示矩阵热图"""
    # attention_array = attention_map_array[0]
    #
    # attention_array = attention_array.reshape(attention_array.shape[0], -1)
    matrices = matrices.unsqueeze(0).unsqueeze(0)
    num_rows, num_cols = matrices.shape[2], matrices.shape[3]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)

    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().cpu().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)


def visualize2(attention_map):
    attention_map_array = attention_map.detach().cpu().numpy()

    attention_array = attention_map_array[0]

    attention_array = attention_array.reshape(attention_array.shape[0], -1)
    input_image = mpimg.imread('/home/maxu/Data/test.data/AFLW2000-3D_crop/image00022.jpg')
    img_height, img_width, _ = input_image.shape
    plt.imshow(input_image)
    plt.imshow(attention_array, cmap='hot', alpha=0.5, interpolation='nearest', extent=[0, img_width, img_height, 0])
    plt.colorbar()
    plt.xlabel('Input')
    plt.ylabel('Output')

    plt.xticks(np.arange(2), ['Input 1', 'Input 2'])
    plt.yticks(np.arange(2), ['Output 1', 'Output 2'])
    plt.axis('off')
    plt.show()


def heat(out):
    img_path = '/HFCAN/benchmark/28.jpg'
    features = out[0]
    # avg_grads = torch.mean(grads[0], dim=(1, 2))
    # avg_grads = avg_grads.expand(features.shape[1], features.shape[2], features.shape[0]).permute(2, 0, 1)
    # features *= avg_grads

    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
    # width = 500
    # height = (img.size[1] * width / img.size[0])
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # heatmap = cv2.resize(heatmap, (width, height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = np.uint8(heatmap * 0.5 + img * 0.5)
    cv2.imshow('a2', superimposed_img)
    cv2.waitKey(0)


class AttentionTransformer(nn.Module):
    def __init__(self, in_channels=256, depth=128):
        super(AttentionTransformer, self).__init__()
        self.aspp = ASPP_CBAM(in_channel=in_channels, depth=depth)
        # self.se = SE_ASPP(in_planes=in_channels // 2, planes=in_channels)
        # self.se = SE_ASPP(in_channels // 2, in_channels)
        # self.aspp = nn.Sequential()
        self.se = SEBlock(in_planes=in_channels // 2, planes=in_channels)

    def forward(self, x):
        feature = self.aspp(x)
        # print(feature.shape)

        # show_heatmaps(feature, 8, 8)
        # visualize2(feature)

        # show_heatmaps(feature, 8, 8)
        # print(feature.shape)
        # visualize1(feature)
        # visualize(feature)
        # visualize2(feature)
        out = self.se(feature)
        # print(out.shape)
        # visualize2(out)

        return out

# def demo():
#     img = torch.randn(1, 768, 8, 8)
#     net = AttentionTransformer()
#     out = net(img)
#     print(out.size())
#
# demo()
