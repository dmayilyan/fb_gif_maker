#/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.misc import imread, imsave, imshow
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import metrics

from urllib import request
import tempfile

class GifItem():
    def __init__(self):
        image = _get_image()

    def _get_image(self):
        filename = './shake.png'
        return imread(filename, mode='RGBA')


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




def _get_image():
    filename = './s_man.png'
    return imread(filename, mode='RGBA')

def _get_array(im):
    im_graph = []
    for (row_y, row) in enumerate(im):
        for (point_x, point) in enumerate(row):
            if point != 0:
                im_graph.append([point_x, row_y])

    im_graph = np.array(im_graph)

    return im_graph

def _get_sils(im_graph):
    sil_score = []
    for n_clust in range(2, 40):
        score = _get_sil_score(n_clust, im_graph)
        sil_score.append([n_clust, score])
        print('Silhouette score of %.4f for %d clusters.' % (score, n_clust))

    return np.array(sil_score)

def _get_sil_score(n_clusters, im_graph):
    kmeans = KMeans(n_clusters=n_clusters, n_jobs = 2)
    kmeans.fit(im_graph)
    y_kmeans = kmeans.predict(im_graph)

    # plt.scatter(im_graph[:, 0], im_graph[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5);
    labels = kmeans.labels_
    sample_size = int(im_graph.size * 0.05)
    return metrics.silhouette_score(im_graph, labels, metric='euclidean', sample_size=sample_size)

def _get_sil_max_pair(sil_scores):
    sc = sil_scores[..., 1]
    arg = np.argmax(sc)

    return sil_scores[arg]

def kmeans():
    image = _get_image()

    # First stage of scan.
    # Overall number of frames
    im = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3

    im_graph = _get_array(im)

    silhouette_scores = _get_sils(im_graph)
    sil_max_pair = _get_sil_max_pair(silhouette_scores)
    all_iamge_count = sil_max_pair[0]

    print('Found %d pictures with score of %f' % (sil_max_pair[0], sil_max_pair[1]) )
    print('Getting number of photos on one row.')


    # Second stage of scan.
    # Number of frames in one row
    im = (image[:80, :, 0] + image[:80, :, 1] + image[:80, :, 2]) / 3
    print(im.size)
    # print(len(im[0]))

    # plt.imshow(im)
    # plt.show()

    im_graph = _get_array(im)

    silhouette_scores = _get_sils(im_graph)
    sil_max_pair = _get_sil_max_pair(silhouette_scores)
    one_row_count = sil_max_pair[0]

    print('Found %d pictures with score of %.4f' %
          (sil_max_pair[0], sil_max_pair[1]) )



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
    # plt.imshow(imread('temp.png'))
    # plt.show()
    kmeans()
    # clusters()
    # test()
