#/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.misc import imread, imsave
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt


def main():
    filename = './shake.png'
    image = imread(filename, mode='RGBA')
    print(image.shape)

    # width = image.shape[0]
    # height = image.shape[1]
    small_size_w = int(image.shape[0] / 3)
    small_size_h = int(image.shape[1] / 3)
    # images = np.zeros(60, 60)
    plt.figure(1)
    plt.imshow(image)
    plt.figure(2)

    counter = 0
    frames = []
    alphas = []
    for i in range(3):
        for j in range(3):
            x = int((i % 3) * small_size_w + small_size_w)
            y = int((j % 3) * small_size_h + small_size_h)

            x0 = x - small_size_w
            y0 = y - small_size_h
            # print(x0, x)
            # print(y0, y, end='\n\n')

            plt.subplot(331 + counter)
            frames.append(image[x0:x, y0:y, :2])
            # alphas.append(image[x0:x, y0:y, 3])
            im = image[x0:x, y0:y, :]
            plt.imshow(im)

            counter += 1

    clip = ImageSequenceClip(frames, fps=15)#, with_mask=True, ismask=alphas)
    clip.write_gif('shake.gif')
    # print(image[:, :, 0])

    # im = image[:80, :80, :]
    # print(im.shape)
    # plt.imshow(im)
    plt.show()


if __name__ == "__main__":
    main()
