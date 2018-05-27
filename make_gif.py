#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# image manipulation and plotting
import numpy as np
from scipy.misc import imread, imsave, imshow
import matplotlib.pyplot as plt

# Clustering
from sklearn.cluster import KMeans
from sklearn import metrics

# Gif creation
from moviepy.editor import ImageSequenceClip

# Image from web
from urllib import request
import tempfile

image = None
all_imege_count = 0
one_row_count = 0
filename = 'cat'

# class GifItem():
#     def __init__(self):
#         image = _get_image()

#     def _get_image(self):
#         filename = './shake.png'
#         return imread(filename, mode='RGBA')


def main():
    '''Main container of gif creation. Needs refactoring.'''
    global image
    global all_imege_count
    global one_row_count
    global filename
    # filename = './s_man.png'
    # image = imread(filename, mode='RGBA')

    # one_row_count = 6
    # all_imege_count = 32
    column_count = int(all_imege_count // one_row_count)
    if all_imege_count % one_row_count != 0:
        column_count += 1
    small_size_w = int(image.shape[0] / one_row_count)
    small_size_h = int(image.shape[1] / column_count)

    # images = np.zeros(60, 60)
    # print(small_size_w, small_size_h)

    # plt.figure(1)
    # plt.imshow(image)
    plt.figure(2)

    counter = 0
    frames = []
    # alphas = []
    for i in range(one_row_count):
        for j in range(column_count):
            x = int((i % one_row_count) * small_size_w + small_size_w)
            y = int((j % column_count) * small_size_h + small_size_h)

            x0 = x - small_size_w
            y0 = y - small_size_h
            # print(x0, x)
            # print(y0, y, end='\n\n')

            # z = np.zeros((x,y))
            # print(z)
            if counter >= all_imege_count:
                continue
            plt.subplot(one_row_count, column_count, counter + 1)
            frames.append(image[x0:x, y0:y, :])
            # alphas.append(z)
            im = image[x0:x, y0:y, :]
            plt.imshow(im)

            counter += 1

    print(len(frames))
    # , with_mask=True, ismask=alphas)
    clip = ImageSequenceClip(frames, fps=10)
    clip.write_gif(filename + '.gif')

    # print(image[:, :, 0])

    # im = image[:80, :80, :]
    # print(im.shape)
    # plt.imshow(im)
    plt.show()


# ========================================================
def _get_image():
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
                # print(sil_score[n_clust - 2][1], sil_score[n_clust - 3][1])
                skip_buffer += 1
                # print(skip_buffer)

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
    image = _get_image()

    # First stage of scan.
    # Overall number of frames

    im = _get_channel_average(image)

    # plt.imshow(im_1)
    # plt.show()

    sil_max_pair = _get_frame_count(im)
    all_imege_count = sil_max_pair[0]

    print('Found %d pictures with score of %f' %
          (sil_max_pair[0], sil_max_pair[1]))
    print('Getting number of photos on one row.')

    # Second stage of scan.
    # Number of frames in one row
    im = _get_channel_average(image, 80)

    sil_max_pair = _get_frame_count(im)
    one_row_count = int(sil_max_pair[0])

    print('Found %d pictures with score of %.4f' %
          (sil_max_pair[0], sil_max_pair[1]))

    # Dimensions of the picture are %dx

    # plt.plot(silhouette_scores)
    # print(kmeans.inertia_)
    # plt.show()


if __name__ == "__main__":
    # url = 'https://scontent-ams3-1.xx.fbcdn.net/v/t39.1997-6/p480x480/10333117_657500967666494_1630318166_n.png?_nc_cat=0&oh=321e4797068402fd69862e12ab4cce2e&oe=5B7672A8'
    # with urllib.request.urlopen(url) as response:
    #     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
    #         shutil.copyfileobj(response, tmp_file)
    # qwe = request.urlretrieve(url, 'temp.png')
    # print(type(qwe))
    # print(qwe)

    kmeans()
    main()

    # clusters()
    # test()
