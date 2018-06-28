import os
import json
import numpy as np
import random

from tqdm import tqdm
import imutil

DATASET_NAME = 'rectangles'
COUNT = 10 * 1000
WIDTH = 64
HEIGHT = 64

os.makedirs('{}/images'.format(DATASET_NAME), exist_ok=True)

examples = []

for i in tqdm(range(COUNT)):
    rect_center_x = np.random.randint(0, WIDTH)
    rect_center_y = np.random.randint(0, HEIGHT)
    rect_radius = np.random.randint(1, 10)
    rect_color = np.random.uniform(size=3) * 255

    pixels = np.zeros((HEIGHT, WIDTH, 3))
    bottom = rect_center_y - rect_radius
    top = rect_center_y + rect_radius
    left = rect_center_x - rect_radius
    right = rect_center_x + rect_radius
    pixels[bottom:top, left:right] = rect_color

    filename = '{}/images/{:09d}.png'.format(DATASET_NAME, i)
    imutil.show(pixels, filename=filename, normalize_color=False)

    examples.append({
        'filename': filename,
        #'label': 0,
    })

with open('{}.dataset'.format(DATASET_NAME), 'w') as fp:
    for example in examples:
        fp.write(json.dumps(example) + '\n')
