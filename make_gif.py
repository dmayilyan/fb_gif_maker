#/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.misc import imread, imsave, imshow
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import metrics


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


def kmeans():
    filename = './s_man.png'
    image = imread(filename, mode='RGBA')
    # print(image[:, :, 0])
    im = image[:80, :, 0]
    print(im.size)
    # print(len(im[0]))

    plt.imshow(im)
    plt.show()

    im_graph = []
    for (row_y, row) in enumerate(im):
        for (point_x, point) in enumerate(row):
            if point != 0:
                # print(point)
                im_graph.append([point_x, row_y])

    im_graph = np.array(im_graph)

    sil_score = []
    for n_clust in range(2, 40):
        print('Silhouette score for {} clusters.'.format(n_clust))
        score = get_sil_score(n_clust, im_graph)
        sil_score.append(score)
        print(score)


    plt.plot(sil_score)
    # print(kmeans.inertia_)
    plt.show()

def get_sil_score(n_clusters, im_graph):
    kmeans = KMeans(n_clusters=n_clusters, n_jobs = 2)
    kmeans.fit(im_graph)
    y_kmeans = kmeans.predict(im_graph)

    # plt.scatter(im_graph[:, 0], im_graph[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5);
    labels = kmeans.labels_

    sample_size = int(im_graph.size * 0.05)
    return metrics.silhouette_score(im_graph, labels, metric='euclidean', sample_size=sample_size)

if __name__ == "__main__":
    kmeans()
    # clusters()
    # test()
