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

# Here is a problem
filename = 's_man.png'


def get_parser(args=None):
    parser = argparse.ArgumentParser(description='Tool to generate .gif file' +
                                     ' from facebook sticker.')

    parser.add_argument('-f', '--file', help='Import image as a file')
    parser.add_argument('-l', '--list', action='store_true',
                        help='List image files in the current dir.')
    parser.add_argument('-u', '--url',
                        help='Get an image from direct web link.')
    parser.add_argument('--fps', nargs='?', const=10, type=int,
                        help='Frame rate of the output gif.')

    return parser


class ImageFrames:
    """Create frames from input image"""
    def __init__(self, image):
        self.image = image
        self.all_imege_count = 0
        self.one_row_count = 0
        self.seperate_frames()

        # super(ImageFrames, self).__init__()
        # self.arg = arg

    def seperate_frames(self):
        # First stage of scan.
        # Overall number of frames
        im = self._get_channel_average()

        sil_max_pair = self._get_frame_count(im)
        self.all_imege_count = sil_max_pair[0]

        print('Found %d pictures with score of %f' %
              (sil_max_pair[0], sil_max_pair[1]))
        print('Getting number of photos on one row.')

        # Second stage of scan.
        # Number of frames in one row
        im = self._get_channel_average(50)
        # plt.imshow(im)
        # plt.show()

        sil_max_pair = self._get_frame_count(im)
        self.one_row_count = int(sil_max_pair[0])

        print('Found %d pictures with score of %.4f' %
              (sil_max_pair[0], sil_max_pair[1]))

    def _get_channel_average(self, y_cut=None):
        '''Average of color channels'''
        return (self.image[:y_cut, :, 0] +
                self.image[:y_cut, :, 1] +
                self.image[:y_cut, :, 2]) / 3

    def _get_frame_count(self, im):
        im_graph = self._get_point_array(im)

        silhouette_scores = self._get_sils(im_graph)
        return self._get_sil_max_pair(silhouette_scores)

    def _get_point_array(self, im):
        '''Get non-zero point array.'''
        im_graph = []
        for (row_y, row) in enumerate(im):
            for (point_x, point) in enumerate(row):
                if point != 0:
                    im_graph.append([point_x, row_y])

        return np.array(im_graph)

    def _get_sils(self, im_graph):
        '''Go through silhouette score counting.'''
        sil_score = []
        # Nah appraoach of skipping extra score calculations
        skip_buffer = 0
        for n_clust in range(2, 40):
            if skip_buffer > 9:
                continue
            score = self._get_sil_score(n_clust, im_graph)
            sil_score.append([n_clust, score])
            if n_clust > 2:
                if sil_score[n_clust - 2][1] < sil_score[n_clust - 3][1]:
                    skip_buffer += 1

            print('Silhouette score of %.4f for %d clusters.' %
                  (score, n_clust))

        return np.array(sil_score)

    def _get_sil_max_pair(self, sil_scores):
        '''Select pair with maximum silhouette score.'''
        sc = sil_scores[..., 1]
        arg = np.argmax(sc)

        return sil_scores[arg]

    def _get_sil_score(self, n_clusters, im_graph):
        '''Get one Silhouette score.'''
        kmeans = KMeans(n_clusters=n_clusters, n_jobs=2)
        kmeans.fit(im_graph)

        # y_kmeans = kmeans.predict(im_graph)
        # plt.scatter(im_graph[:, 0], im_graph[:, 1],
        #             c=y_kmeans, s=50, cmap='viridis')

        # centers = kmeans.cluster_centers_
        # plt.scatter(centers[:, 0], centers[:, 1],
        #             c='black', s=100, alpha=0.5)

        labels = kmeans.labels_
        sample_size = int(im_graph.size * 0.05)
        return metrics.silhouette_score(im_graph, labels,
                                        metric='euclidean',
                                        sample_size=sample_size)


class GifItem(ImageFrames):
    def __init__(self, image):
        ImageFrames.__init__(self, image)
        self.column_count = self._count_column()
        self._make_frames()

    def _count_column(self):
        '''Main container of gif creation. Needs refactoring.'''

        column_count = int(self.all_imege_count // self.one_row_count)
        print(self.all_imege_count % self.one_row_count)
        if self.all_imege_count % self.one_row_count != 0:
            column_count += 1

        return column_count


    def _make_frames(self):
        small_size_w = int(self.image.shape[1] / self.one_row_count)
        small_size_h = int(self.image.shape[0] / self.column_count)

        plt.figure(1)
        plt.imshow(self.image)
        plt.show()
        plt.figure(2)

        # alphas = []
        print('Width %d\tHeight: %d' % (self.image.shape[1],
                                        self.image.shape[0]))
        print('One row: %d\tColumn_count: %d\n' % (self.one_row_count,
                                                   self.column_count))

        counter = 0
        frames = []
        for i in range(self.column_count):
            for j in range(self.one_row_count):
                x = int((j % self.one_row_count) * small_size_w + small_size_w)
                y = int((i % self.column_count) * small_size_h + small_size_h)

                # print('X: %d\t Y: %d' % (x, y))

                x0 = x - small_size_w
                y0 = y - small_size_h
                # print(x0, x)
                # print(y0, y, end='\n\n')

                # z = np.zeros((x,y))
                # print(z)
                if counter >= self.all_imege_count:
                    continue
                plt.subplot(self.column_count,
                            self.one_row_count,
                            counter + 1)

                frames.append(self.image[y0:y, x0:x, :])
                # alphas.append(z)
                im = self.image[y0:y, x0:x, :]
                # imsave("./tempdir/frame_%d.png" % counter, im)
                plt.imshow(im)
                # plt.show()

                counter += 1

        self._make_gif(frames)

        plt.show()

    def _make_gif(self, frames):
        clip = ImageSequenceClip(frames, fps=10)
        # clip = ImageSequenceClip('./tempdir/', fps=10, with_mask=True)
        clip.write_gif(filename + '.gif')


def _get_file_image(fn):
    '''Read image from file.'''
    # global filename
    fn = './' + fn
    return imread(fn, mode='RGBA')

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


def handle_args(args=None):
    if args is None:
        sys.exit('No image was given.')

    arg_list = get_parser().parse_args(args)

    if arg_list.list:
        files = []
        [files.append(f) for f in os.listdir('./') if f.endswith('.png')]
        if files != []:
            print('Available supported images in the current dir are:')
            for (i, f) in enumerate(files):
                print('%d. %s' % (i + 1, f))
        else:
            print('There are no .png files in the dir.')

    if arg_list.file:
        if os.path.isfile(arg_list.file):
            print(arg_list.file)
            im = _get_file_image(arg_list.file)
            return im
            # plt.imshow(im)
            # plt.show()
        else:
            print('File not found.')

    if arg_list.url:
        im = _get_url_image(arg_list.url)
        return im


def main(arguments):
    image = handle_args(arguments)

    GifItem(image)


if __name__ == "__main__":
    # handle_args(sys.argv[1:])

    # im_frames = ImageFrames()
    # im_frames.seperate_frames()

    main(sys.argv[1:])
