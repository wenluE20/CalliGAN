# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

from PIL import Image
import numpy as np
from io import StringIO


def pad_seq(seq, batch_size):
    # pad the sequence to be the multiples of batch_size
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq


def bytes_to_file(bytes_img):
    return StringIO(bytes_img)


def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized


def read_split_image(img):
    # 使用PIL替代scipy.misc.imread
    with Image.open(img) as pil_img:
        # 转换为灰度并获取数组
        mat = np.array(pil_img.convert('L')).astype(np.float)
    side = int(mat.shape[1] / 2)
    assert side * 2 == mat.shape[1]
    img_A = mat[:, :side]  # target
    img_B = mat[:, side:]  # source

    return img_A, img_B


def read_split_image_rgb(img):
    # 使用PIL替代scipy.misc.imread
    with Image.open(img) as pil_img:
        mat = np.array(pil_img).astype(np.float)
    side = int(mat.shape[1] / 2)
    assert side * 2 == mat.shape[1]
    img_A = mat[:, :side]  # target
    img_B = mat[:, side:]  # source

    return img_A, img_B


def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    # 使用PIL替代scipy.misc.imresize
    w, h = img.shape
    # 转换为PIL图像
    pil_img = Image.fromarray(img.astype(np.uint8))
    # 调整大小
    enlarged = np.array(pil_img.resize((nh, nw), Image.LANCZOS)).astype(np.float)
    return enlarged[shift_x:shift_x + w, shift_y:shift_y + h]


def shift_and_resize_image_rgb(img, shift_x, shift_y, nw, nh):
    # 使用PIL替代scipy.misc.imresize
    w, h, _ = img.shape
    # 转换为PIL图像
    pil_img = Image.fromarray(img.astype(np.uint8))
    # 调整大小
    enlarged = np.array(pil_img.resize((nh, nw), Image.LANCZOS)).astype(np.float)
    return enlarged[shift_x:shift_x + w, shift_y:shift_y + h]


def scale_back(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def save_concat_images(imgs, img_path):
    # 使用PIL替代scipy.misc.imsave
    concated = np.concatenate(imgs, axis=1)
    # 如果是浮点数组且范围在[-1,1]，先转换到[0,255]
    if np.min(concated) < 0 or np.max(concated) > 1:
        concated = scale_back(concated) * 255
    # 转换为PIL图像并保存
    Image.fromarray(concated.astype(np.uint8)).save(img_path)


"""
def compile_frames_to_gif(frame_dir, gif_file):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    print(frames)
    images = [misc.imresize(imageio.imread(f), interp='nearest', size=0.33) for f in frames]
    imageio.mimsave(gif_file, images, duration=0.1)
    return gif_file
"""
