from matplotlib import pyplot as plt
import cv2
import json
import os

base = '../deskewed'

with open('pages.json', 'r') as f:
    pages = json.load(f)

for i, page in enumerate(pages):
    if i < 1:
        print page
        # img = cv2.imread(os.path.join(base, page), 1)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        for j, row in enumerate(pages[page]):
            if j < 1:
                for col in pages[page][row]:
                    img = cv2.imread(pages[page][row][col], 0)
                    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
                    plt.show()
                    # cv2.imshow('image', img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

            # img = cv2.imread(pages[page][row]['5'], 1)
            # cv2.imshow('image', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

exit(0)
