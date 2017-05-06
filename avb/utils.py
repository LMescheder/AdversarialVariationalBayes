"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import numpy as np
import scipy as sp
import scipy.misc
from time import gmtime, strftime
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, output_dir, image_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return imsave(inverse_transform(images), size, os.path.join(output_dir, image_name))


def save_kde_plot(z, output_dir, image_name, bbox=[-5, 5, -5, 5]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x = z[:, 0]
    y = z[:, 1]
    values = np.vstack([x, y])
    kernel = sp.stats.gaussian_kde(values)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis(bbox)

    xx, yy = np.mgrid[bbox[0]:bbox[1]:100j, bbox[2]:bbox[3]:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap='Blues')
    cset = ax.contour(xx, yy, f, colors='k')
    # ax.plot(z[:, 0], z[:, 1], 'x')

    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight')
    plt.close(fig)

def save_heat_map(f, output_dir, image_name, samples=None, bbox=[-5, 5, -5, 5]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    N, M = f.shape
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis(bbox)

    # ax.imshow(f, cmap='hot',
    #     origin='lower',
    #     extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
    xx, yy = np.mgrid[bbox[0]:bbox[1]:(N*1j), bbox[2]:bbox[3]:(M*1j)]
    cfset = ax.contourf(xx, yy, f, cmap='Reds')
    cset = ax.contour(xx, yy, f, colors='k')

    if samples is not None:
        ax.plot(samples[:, 0], samples[:, 1], 'x')

    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight')
    plt.close(fig)

def save_z_plot(z, zlabels, output_dir, image_name, bbox=[-5, 5, -5, 5]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig, ax = plt.subplots()

    ax.set_autoscale_on(False)
    for label in set(zlabels):
        z_i = z[zlabels==label]
        ax.plot(z_i[:, 0], z_i[:, 1], 'x')

    ax.set_aspect("equal")
    ax.axis(bbox)

    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight')
    plt.close(fig)

def get_bbox(samples):
    min_x, min_y = np.min(samples, axis=0)
    max_x, max_y = np.max(samples, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    center_x = (max_x + min_x)/2
    center_y = (max_y + min_y)/2
    half_size = 0.55 * max(width, height)

    bbox = [
        center_x - half_size, center_x + half_size,
        center_y - half_size, center_y + half_size
    ]

    return bbox

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    #return scipy.misc.imsave(path, merge(images, size))
    return scipy.misc.toimage(merge(images, size), cmin=0., cmax=1.).save(path)

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    # return np.array(cropped_image)/127.5 - 1.
    return  np.array(cropped_image)/255.

def inverse_transform(images):
    return images
    # return (images + 1.)/2

def to_nested_dict(d):
    nested_d = defaultdict(dict)
    for (k1, k2), v in d.items():
        nested_d[k1][k2] = v
    return nested_d

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict
