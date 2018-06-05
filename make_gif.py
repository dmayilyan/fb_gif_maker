#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blablabla
"""

# General purpose
import sys
import os


# image manipulation and plotting
import numpy as np
from scipy.misc import imread, imsave, imshow
import matplotlib.pyplot as plt
import argparse

# Clustering
from sklearn.cluster import KMeans
from sklearn import metrics

# Gif creation
from moviepy.editor import ImageSequenceClip

# Image from web
from urllib import request
# from urllib2 import Request, urlopen, URLError
import tempfile
import shutil


image = None
all_imege_count = 0
one_row_count = 0
filename = 's_man'


def get_parser(args=None):
    parser = argparse.ArgumentParser(description='Tool to generate .gif file' +
                                     ' from facebook sticker.')

    parser.add_argument('-f', '--file', help='Import image as a file')
    parser.add_argument('-l', '--list', action='store_true',
                        help='List image files in the current dir.')
    parser.add_argument('-u', '--url',
                        help='Get an image from direct web link.')

    return parser


# class GifItem():
#     def __init__(self):
#         image = _get_file_image()

#     def _get_file_image(self):
#         filename = './shake.png'
#         return imread(filename, mode='RGBA')


def main1():
    '''Main container of gif creation. Needs refactoring.'''
    global image
    global all_imege_count
    global one_row_count
    global filename
    image = imread('./' + filename + '.png', mode='RGBA')

    # one_row_count = 2
    # all_imege_count = 4
    column_count = int(all_imege_count // one_row_count)
    print(column_count)
    # print(all_imege_count % one_row_count)
    if all_imege_count % one_row_count != 0:
        column_count += 1

    print(column_count)
    # return 0
    print(image.shape)
    small_size_w = int(image.shape[1] / one_row_count)
    small_size_h = int(image.shape[0] / column_count)

    # images = np.zeros(60, 60)
    print(small_size_w, small_size_h)

    # plt.figure(1)
    # plt.imshow(image)
    plt.figure(2)

    counter = 0
    frames = []
    # alphas = []
    print('Width %d\tHeight: %d' % (image.shape[1], image.shape[0]))
    print('One row: %d\tColumn_count: %d\n' % (one_row_count, column_count))

    for i in range(column_count):
        for j in range(one_row_count):
            x = int((j % one_row_count) * small_size_w + small_size_w)
            y = int((i % column_count) * small_size_h + small_size_h)

            # print('X: %d\t Y: %d' % (x, y))

            x0 = x - small_size_w
            y0 = y - small_size_h
            # print(x0, x)
            # print(y0, y, end='\n\n')

            # z = np.zeros((x,y))
            # print(z)
            if counter >= all_imege_count:
                continue
            plt.subplot(column_count, one_row_count, counter + 1)

            # Here is a bug
            # x,y = y,x
            # x0,y0 = y0,x0
            frames.append(image[y0:y, x0:x, :])
            # alphas.append(z)
            im = image[y0:y, x0:x, :]
            # imsave("./tempdir/frame_%d.png" % counter, im)
            plt.imshow(im)
            # plt.show()

            counter += 1

    clip = ImageSequenceClip(frames, fps=10)
    # clip = ImageSequenceClip('./tempdir/', fps=10, with_mask=True)
    clip.write_gif(filename + '.gif')

    # print(image[:, :, 0])

    # im = image[:80, :80, :]
    # print(im.shape)
    # plt.imshow(im)
    plt.show()


# ========================================================
def _get_file_image():
    '''Read image from file.'''
    global filename
    fn = './' + filename + '.png'
    return imread(fn, mode='RGBA')


def _get_point_array(im):
    '''Get non-zero point array.'''
    im_graph = []
    for (row_y, row) in enumerate(im):
        for (point_x, point) in enumerate(row):
            if point != 0:
                im_graph.append([point_x, row_y])

    im_graph = np.array(im_graph)

    return im_graph


def _get_sils(im_graph):
    '''Go through silhouette score counting.'''
    sil_score = []
    # Nah appraoach of skipping extra score calculations
    skip_buffer = 0
    for n_clust in range(2, 40):
        if skip_buffer > 9:
            continue
        score = _get_sil_score(n_clust, im_graph)
        sil_score.append([n_clust, score])
        if n_clust > 2:
            if sil_score[n_clust - 2][1] < sil_score[n_clust - 3][1]:
                skip_buffer += 1

        print('Silhouette score of %.4f for %d clusters.' % (score, n_clust))

    return np.array(sil_score)


def _get_sil_score(n_clusters, im_graph):
    '''Get one Silhouette score.'''
    kmeans = KMeans(n_clusters=n_clusters, n_jobs=2)
    kmeans.fit(im_graph)

    # y_kmeans = kmeans.predict(im_graph)
    # plt.scatter(im_graph[:, 0], im_graph[:, 1],
    #             c=y_kmeans, s=50, cmap='viridis')

    # centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5);
    labels = kmeans.labels_
    sample_size = int(im_graph.size * 0.05)
    return metrics.silhouette_score(im_graph, labels,
                                    metric='euclidean',
                                    sample_size=sample_size)


def _get_sil_max_pair(sil_scores):
    '''Select pair with maximum silhouette score.'''
    sc = sil_scores[..., 1]
    arg = np.argmax(sc)

    return sil_scores[arg]


def _get_frame_count(im):
    im_graph = _get_point_array(im)

    silhouette_scores = _get_sils(im_graph)
    return _get_sil_max_pair(silhouette_scores)


def _get_channel_average(image, y_cut=None):
    '''Average of color channels'''
    return (image[:y_cut, :, 0] +
            image[:y_cut, :, 1] +
            image[:y_cut, :, 2]) / 3


def kmeans():
    global image
    global all_imege_count
    global one_row_count
    image = _get_file_image()

    # First stage of scan.
    # Overall number of frames
    im = _get_channel_average(image)

    sil_max_pair = _get_frame_count(im)
    all_imege_count = sil_max_pair[0]

    print('Found %d pictures with score of %f' %
          (sil_max_pair[0], sil_max_pair[1]))
    print('Getting number of photos on one row.')

    # Second stage of scan.
    # Number of frames in one row
    im = _get_channel_average(image, 80)
    # plt.imshow(im)
    # plt.show()

    sil_max_pair = _get_frame_count(im)
    one_row_count = int(sil_max_pair[0])

    print('Found %d pictures with score of %.4f' %
          (sil_max_pair[0], sil_max_pair[1]))


# ========================================================
def _get_url_image(url):
    # url = 'https://scontent-ams3-1.xx.fbcdn.net/v/t39.1997-6/p480x480/10333117_657500967666494_1630318166_n.png?_nc_cat=0&oh=321e4797068402fd69862e12ab4cce2e&oe=5B7672A8'
    im = ''
    with request.urlopen(url) as response:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            shutil.copyfileobj(response, tmp_file)
            im = imread(tmp_file, mode='RGBA')
            # plt.imshow(im)
            # plt.show()
    return im
    # qwe = request.urlretrieve(url, 'temp.png')


class ImageFrames():
    """Create frames from input image"""
    def __init__(self):
        print(arg_list)
        # super(ImageFrames, self).__init__()
        # self.arg = arg



# def validateUrl(url):
#     # req = Request(url)
#     try:
#         request.urlopen(url)
#     except IOError as e:
#         raise e
#         print('Njah')
#         return False

#     return True


def main(args=None):
    arg_list = get_parser().parse_args(args)

    if arg_list.list:
        files = []
        [files.append(f) for f in os.listdir('./') if f.endswith('.png')]
        if files != []:
            print('Available supported images in the current dir are:')
            [print('%d. %s' % (i + 1, f)) for (i, f) in enumerate(files)]
        else:
            print('There are no .png files in the dir.')

    if arg_list.file:
        if os.path.isfile(arg_list.file):
            print(arg_list.file)
        else:
            print('File not found.')


    # url = 'https://scontent-ams3-1.xx.fbcdn.net/v/t39.1997-6/p480x480/10333117_657500967666494_1630318166_n.png?_nc_cat=0&oh=321e4797068402fd69862e12ab4cce2e&oe=5B7672A8'

    if arg_list.url:
        im = _get_url_image(arg_list.url)
        return im
        # plt.imshow(im)
        # plt.show()

        # print(validateUrl(url))



    # print(arg_list.list)
    # print(arg_list.file)
    # print(arg_list.url)

    # image = ImageFrames()


if __name__ == "__main__":
    handle_args(sys.argv[1:])

    # _get_url_image()
    # kmeans()
    # main()

    # clusters()
