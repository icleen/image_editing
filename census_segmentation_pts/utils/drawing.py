import numpy as np
import cv2
import math

def draw_sol_torch(predictions, org_img, conf_threshold=0.1):
    for j in xrange(predictions.size(1)):

        conf = predictions[0,j,0]
        conf = conf.data.cpu().numpy()[0]
        if conf < 0.1:
            continue

        color = int(255*conf)

        pt0 = predictions[0,j,1:3]# * 512
        pt1 = predictions[0,j,3:5]# * 512

        pt0 = tuple(pt0.data.cpu().numpy().astype(np.int64).tolist())
        pt1 = tuple(pt1.data.cpu().numpy().astype(np.int64).tolist())

        x0,y0 = pt0
        x1,y1 = pt1

        dx = x0-x1
        dy = y0-y1

        mx = (x0+x1)/2.0
        my = (y0+y1)/2.0

        mx = int(mx)
        my = int(my)

        cv2.circle(org_img, (mx, my), 3, color, -1)
    return org_img
