import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import json

img = cv2.imread(sys.argv[1])
with open(sys.argv[2]) as f:
    data = json.load(f)

for pt in data['corners']:
    cv2.circle(img, tuple(map(int,pt)), 10, (0,0,255), -1)

plt.imshow(img)
plt.show()
