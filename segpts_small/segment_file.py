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

from collections import defaultdict

def proj_nms_single(confidence, x_points, overlap_thresh):
    # Based on https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # Maybe could port to pytorch to work over the tensors directly

    idxs = np.argsort(confidence)
    clusters = []

    pick = []
    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]

        # xx is all of the x points that have a lower cnfidence than the current x we are looking at
        xx = x_points[idxs[:last+1]]
        this_x = x_points[i]

        # the list of distances between the current x and all of the other x's with lower confidence`
        dis = np.abs(xx - this_x)
        # gets all of the xx points within a certain distance of the current x
        matches = np.where(dis < overlap_thresh)[0]
        matched_idxs =  idxs[matches]
        # takes all of the 'close' points and clusters them into a list and adds that list to the clusters list
        clusters.append(matched_idxs)
        idxs = np.delete(idxs, matches)

    # returns a list of lists of points, so that each cluster in clusters
    # conatains a bunch of points that are close to each other
    return clusters


def get_img(image_path):
    org_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    rescale_range = config['pretraining']['sol']['validation_rescale_range']
    target_dim1 = rescale_range[0]

    s = target_dim1 / float(org_img.shape[1])
    target_dim0 = int(org_img.shape[0]/float(org_img.shape[1]) * target_dim1)
    full_res_img = org_img
    org_img = cv2.resize(org_img,(target_dim1, target_dim0), interpolation = cv2.INTER_CUBIC)
    return full_res_img, org_img, s


def get_corners(sml_img, sol, s, c_img=None):
    img = sml_img.transpose([2,1,0])[None,...]
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    img = img / 128.0 - 1.0

    img = Variable(img, requires_grad=False, volatile=True).cuda()

    predictions = sol(img)
    predictions = predictions.data.cpu().numpy()
    predictions = predictions[predictions[:,:,0] > 0.1]

    # predictions is a matrix of 2000ish x 5
    # where the first is the confidence and then there are two pairs of x and y coordinates
    # here we are extracting just the first pair since the second is probably a repeat
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
    # make sure there are at least vert_cluster_cnt (41) in vert_clusters
    assert len(vert_clusters) >= vert_cluster_cnt and len(horz_clusters) >= horz_cluster_cnt

    # sort the clusters by how many points are in each cluster, with larger clusters are first
    vert_clusters.sort(key=lambda x:len(x), reverse=True)
    # take the top vert_cluster_cnt (41) clusters that have the most points
    vert_clusters = np.array(vert_clusters[:vert_cluster_cnt])

    horz_clusters.sort(key=lambda x:len(x), reverse=True)
    horz_clusters = np.array(horz_clusters[:horz_cluster_cnt])

    # medians as center of each cluster
    vert_cluster_medians = np.array([np.median(vert_projection[c]) for c in vert_clusters])
    horz_cluster_medians = np.array([np.median(horz_projection[c]) for c in horz_clusters])

    # sort the points
    vert_mean_sort_idx = vert_cluster_medians.argsort()
    horz_mean_sort_idx = horz_cluster_medians.argsort()

    vert_cluster_medians = vert_cluster_medians[vert_mean_sort_idx]
    horz_cluster_medians = horz_cluster_medians[horz_mean_sort_idx]

    # sorts the clusters, according to the median points
    vert_clusters = vert_clusters[vert_mean_sort_idx]
    horz_clusters = horz_clusters[horz_mean_sort_idx]

    output_grid = np.full((vert_cluster_cnt, horz_cluster_cnt, 2), np.nan)

    color = (255,0,0) # red
    MATCH_THRESHOLD = 20.0
    for i, c in enumerate(vert_clusters):

        these_predictions = predictions[c]
        these_confidences = confidence[c]

        vert_median = vert_cluster_medians[i]

        # gets a set of default values for the pts along the x axis
        pts = np.concatenate((horz_cluster_medians[:,None],
            np.full_like(horz_cluster_medians[:,None], vert_median)), axis=1)

        # distances from approximate cluster centers to the actual predictions in that cluster
        dis = (these_predictions[:,None,:]-pts[None,:,:])**2
        dis = np.sqrt(dis.sum(axis=-1))
        # 41 x 1 vector
        min_dis_idx = dis.argmin(axis=1)
        min_dis = dis.min(axis=1)

        # probably a non-for-loop way to do this
        # only pick a point if it is within MATCH_THRESHOLD of the cluster center
        d = defaultdict(lambda: (MATCH_THRESHOLD, None))
        for pt_idx, j, conf in zip(np.arange(len(min_dis_idx)), min_dis_idx, min_dis):

            prev_score, _ = d[j]
            # for each cluster you pick the smallest point that is within the MATCH_THRESHOLD
            if prev_score > conf:
                d[j] = conf, c[pt_idx]

        # this just fills in the grid with the points selected previously
        for j, a in d.iteritems():
            output_grid[i,j] = predictions[a[1]]

        if c_img is not None:
            for j,pt in enumerate(these_predictions):
                x = int(pt[0])
                y = int(pt[1])

                cv2.circle(c_img, (x, y), 5, color, 1)

    # if there was not a point within the MATCH_THRESHOLD in the previous step,
    # then the point in the grid will still be nan, meaning that we couldn't
    # find a good point for that grid spot
    nan_idxs = np.where(np.isnan(output_grid[:,:,0]))


# for each nan point, we walk right, left, up and down until we either find a point or hit the edge
# if we get a right and left, we average those; if we get just a right or left, we take the existing one
# if we get none then we throw an error (the assertion below)
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

        # This doees the averaging part described above
        output_grid[i,j,0] = (output_grid[i_pos, j,0] + output_grid[i_neg, j,0])/2.0
        output_grid[i,j,1] = (output_grid[i, j_pos,1] + output_grid[i, j_neg,1])/2.0

    return output_grid, c_img


def write_segments(corners, img, img_path):
    if not os.path.exists("segments"):
        os.makedirs("segments")

    part_path = img_path.split('/')[-1].split('.')[0]
    page = {}


    for i in range(corners.shape[0]-1):
        person = {}
        for j in range(corners.shape[1]-1):
            pt0 = corners[i,j] / s
            pt1 = corners[i+1,j] / s
            pt2 = corners[i,j+1] / s
            pt3 = corners[i+1,j+1] / s

            min_x = int(min([pt0[0], pt1[0], pt2[0], pt3[0]]))
            max_x = int(max([pt0[0], pt1[0], pt2[0], pt3[0]]))

            min_y = int(min([pt0[1], pt1[1], pt2[1], pt3[1]]))
            max_y = int(max([pt0[1], pt1[1], pt2[1], pt3[1]]))

            crop = img[min_y:max_y, min_x:max_x]

            cv2.imwrite("segments/{}_{}_{}.png".format(part_path,i,j), crop)
            person[j] = "segments/{}_{}_{}.png".format(part_path,i,j)
        page[i] = person

    return page


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print "Usage: ", sys.argv[0], " yaml_config_file image_path"
        sys.exit()

    with open(sys.argv[1]) as f:
        config = yaml.load(f)

    sol_network_config = config['network']['sol']
    pretrain_config = config['pretraining']

    sol = continuous_state.init_model(config)

    image_file = sys.argv[2]
    big_img, sml_img, s = get_img(image_file)
    corners, c_img = get_corners(sml_img, sol, s, c_img=big_img.copy())

    color = (0,0,255) # blue
    for i in range(output_grid.shape[0]):
        for j in range(output_grid.shape[1]):

            pt = output_grid[i,j]

            x = int(pt[0])
            y = int(pt[1])
            cv2.circle(c_img, (x, y), 5, color, 1)

    part_path = img_path.split('/')[-1].split('.')[0]
    cv2.imwrite("{}_visual.png".format(part_path), c_img)

    page = {image_file: write_segments(corners, big_img, f)}
    i += 1

    with open('page.json', 'w') as f:
        json.dump(page, f)

#
#
#
#
#
#
#
#
#
#
#
