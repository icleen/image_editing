
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils import continuous_state
import numpy as np
import cv2
import json
import yaml
import sys
import os
import math

from utils import transformation_utils, drawing
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import KMeans

def sol_non_max_suppression(start_torch, overlap_thresh):
    start = start_torch.data.cpu().numpy()

    pick = sol_nms_single(start[0], overlap_thresh)

    zero_idx = [0 for _ in xrange(len(pick))]

    select = (zero_idx, pick)
    return start_torch[select][None,...]

def proj_nms_single(c,x, overlap_thresh):
    # Based on https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # Maybe could port to pytorch to work over the tensors directly

    idxs = np.argsort(c)
    clusters = []

    pick = []
    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]

        xx = x[idxs[:last+1]]
        this_x = x[i]

        dis = np.abs(xx - this_x)
        matches = np.where(dis < overlap_thresh)[0]
        matched_idxs =  idxs[matches]
        clusters.append(matched_idxs)
        idxs = np.delete(idxs, matches)

    return clusters


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: ", sys.argv[0], " yaml_config_file image_path"
        sys.exit()

    with open(sys.argv[1]) as f:
        config = yaml.load(f)

    sol_network_config = config['network']['sol']
    pretrain_config = config['pretraining']

    sol = continuous_state.init_model(config)

    img_path = sys.argv[2]

    org_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    rescale_range = config['pretraining']['sol']['validation_rescale_range']
    target_dim1 = rescale_range[0]

    s = target_dim1 / float(org_img.shape[1])
    target_dim0 = int(org_img.shape[0]/float(org_img.shape[1]) * target_dim1)
    full_res_img = org_img
    org_img = cv2.resize(org_img,(target_dim1, target_dim0), interpolation = cv2.INTER_CUBIC)

    img = org_img.transpose([2,1,0])[None,...]
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    img = img / 128.0 - 1.0

    img = Variable(img, requires_grad=False, volatile=True).cuda()

    predictions = sol(img)
    predictions = predictions.data.cpu().numpy()
    predictions = predictions[predictions[:,:,0] > 0.1]

    confidence = predictions[:,0]
    predictions = predictions[:,1:3]


    vertical_axis = np.array([
    0,1
    ])

    horizontal_axis = np.array([
    1,0
    ])

    vert_cluster_cnt = 41
    vert_projection = np.matmul(predictions, vertical_axis)[:,None]

    horz_cluster_cnt = 41
    horz_projection = np.matmul(predictions, horizontal_axis)[:,None]

    vert_clusters = proj_nms_single(confidence, vert_projection, 10.0)
    horz_clusters = proj_nms_single(confidence, horz_projection, 10.0)

    vert_clusters.sort(key=lambda x:len(x), reverse=True)
    vert_clusters = np.array(vert_clusters[:vert_cluster_cnt])

    horz_clusters.sort(key=lambda x:len(x), reverse=True)
    horz_clusters = np.array(horz_clusters[:horz_cluster_cnt])

    vert_cluster_means = np.array([np.median(vert_projection[c]) for c in vert_clusters])
    horz_cluster_means = np.array([np.median(horz_projection[c]) for c in horz_clusters])

    vert_mean_sort_idx = vert_cluster_means.argsort()
    horz_mean_sort_idx = horz_cluster_means.argsort()

    vert_cluster_means = vert_cluster_means[vert_mean_sort_idx]
    horz_cluster_means = horz_cluster_means[horz_mean_sort_idx]

    vert_clusters = vert_clusters[vert_mean_sort_idx]
    horz_clusters = horz_clusters[horz_mean_sort_idx]


    output_grid = np.full((vert_cluster_cnt, horz_cluster_cnt, 2), np.nan)

    nan_idxs = np.where(np.isnan(output_grid[:,:,0]))

    for i,j in zip(*nan_idxs):

        i_neg = i-1
        while i_neg > -1 and np.isnan(output_grid[i_neg, j, 0]):
            i_neg -= 1
        i_neg = None if i_neg == -1 else i_neg

        i_pos = i+1
        while i_pos < output_grid.shape[0] and np.isnan(output_grid[i_pos, j, 0]):
            i_pos += 1
        i_pos = None if i_pos == output_grid.shape[0] else i_pos


        j_neg = j-1
        while j_neg > -1 and np.isnan(output_grid[i, j_neg, 0]):
            j_neg -= 1
        j_neg = None if j_neg == -1 else j_neg

        j_pos = j+1
        while j_pos < output_grid.shape[1] and np.isnan(output_grid[i, j_pos, 0]):
            j_pos += 1
        j_pos = None if j_pos == output_grid.shape[0] else j_pos


        assert i_pos is not None and i_neg is not None
        if i_pos is None:
            i_pos = i_neg

        if i_neg is None:
            i_net = i_pos

        if j_pos is None:
            j_pos = j_neg

        if j_neg is None:
            j_net = j_pos

        # Just a heads up...
        output_grid[i,j,0] = (output_grid[i_pos, j,0] + output_grid[i_neg, j,0])/2.0
        output_grid[i,j,1] = (output_grid[i, j_pos,1] + output_grid[i, j_neg,1])/2.0


    if not os.path.exists("result"):
        os.makedirs("result")
        os.makedirs("result/cells")

    color = (0,0,255)
    for i in range(output_grid.shape[0]):
        for j in range(output_grid.shape[1]):

            pt = output_grid[i,j]

            x = int(pt[0])
            y = int(pt[1])

            cv2.circle(c_img, (x, y), 5, color, 1)

    cv2.imwrite("result/visual.png", c_img)

    for i in range(output_grid.shape[0]-1):
        for j in range(output_grid.shape[1]-1):
            pt0 = output_grid[i,j] / s
            pt1 = output_grid[i+1,j] / s
            pt2 = output_grid[i,j+1] / s
            pt3 = output_grid[i+1,j+1] / s

            min_x = int(min([pt0[0], pt1[0], pt2[0], pt3[0]]))
            max_x = int(max([pt0[0], pt1[0], pt2[0], pt3[0]]))

            min_y = int(min([pt0[1], pt1[1], pt2[1], pt3[1]]))
            max_y = int(max([pt0[1], pt1[1], pt2[1], pt3[1]]))


            crop = full_res_img[min_y:max_y, min_x:max_x]

            cv2.imwrite("result/cells/{}_{}.png".format(i,j), crop)
