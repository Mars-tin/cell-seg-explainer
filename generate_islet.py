import os
import random
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle


CELLTYPE_COLOR = {
    'alpha' : 'r',
    'beta'  : 'y',
    'delta' : 'b'
}


class Cell:

    def __init__(self, cell_type, x, y, r):

        self.cell_type = cell_type
        self.x = x
        self.y = y
        self.r = r

    def visualize(self):
        return Rectangle((self.x-self.r, self.y-self.r), 2 * self.r, 2 * self.r,
                         facecolor=CELLTYPE_COLOR[self.cell_type], alpha=1)


class Islet:

    def __init__(self, is_t2d, islet_id):

        self.t2d = is_t2d
        self.r = 0
        self.id = islet_id
        self.cells = []
        self.amyloids = []

    def __sizeof__(self):
        return self.r

    def __len__(self):
        return len(self.cells)

    def add_cell(self, cell: Cell):
        self.cells.append(cell)
        r_max = max(np.abs(cell.x), np.abs(cell.y))
        self.r = max(self.r, r_max+cell.r)

    def visualize(self, figsize=(15, 15), save=None):

        assert save in {None, 'png', 'jpg', 'pdf'}

        fig, ax = plt.subplots(figsize=figsize)

        boxes = {}
        handles = []
        for cell_type, color in CELLTYPE_COLOR.items():
            boxes[cell_type] = []
            handles.append(Patch(color=color, label=cell_type))

        for cell in self.cells:
            rect = cell.visualize()
            boxes[cell.cell_type].append(rect)

        for box in boxes.values():
            pc = PatchCollection(box, match_original=True)
            ax.add_collection(pc)

        r = self.r * 1.2
        plt.xlim(-r, r)
        plt.ylim(-r, r)
        ax.legend(handles=handles, loc='upper left')

        if save is not None:
            plt.savefig('type={}_id={}.{}'.format(
                'T2D' if self.t2d else 'ND', self.id, save))
        plt.show()

    def save(self, save='txt'):

        filename = 'type={}_id={}.{}'.format(
                'T2D' if self.t2d else 'ND', self.id, save)
        with open(filename, 'w') as f:
            f.write('Cell ID,XMin,XMax,YMin,YMax,α cell,δ cell,β cell\n')
            for i, cell in enumerate(self.cells):
                x, y, r = cell.x, cell.y, cell.r
                cell_type = [0, 0, 0]
                if cell.cell_type == 'alpha':
                    cell_type[0] = 1
                elif cell.cell_type == 'delta':
                    cell_type[1] = 1
                elif cell.cell_type == 'beta':
                    cell_type[2] = 1
                f.write('{},{},{},{},{},{},{},{}\n'
                        .format(i, x-r, x+r, y-r, y+r,
                                cell_type[0], cell_type[1], cell_type[2]))


def generate_islet(is_t2d, seed=0):

    # Parameters
    islet_min, islet_max = 75, 200
    cell_min, cell_max = 5, 15
    amyloid_min, amyloid_mu, amyloid_max = 30, 40, 50
    cell_mu, cell_sigma = 10, 1

    islet = Islet(is_t2d=is_t2d, islet_id=seed)
    print('Generating Islet id={}, type: {}'
          .format(seed, 'T2D' if is_t2d else 'ND'))

    # Set up random states
    random.seed(seed)
    np.random.seed(seed)
    rs = np.random.RandomState(seed=seed)

    cell_min_dist = rs.uniform(cell_min * 2, cell_max)
    density_beta = cell_max / cell_min_dist
    thickness_delta = rs.uniform(1.2, 1.5)
    thickness_alpha = rs.uniform(1.5, 1.75)

    islet_radius_x, islet_radius_y = rs.uniform(islet_min, islet_max, 2)
    print('Size (x, y): {}, {}'
          .format(islet_radius_x, islet_radius_y))

    r_generate = max(islet_radius_x, islet_radius_y) * thickness_alpha
    num_beta = int(np.square(r_generate / cell_mu) * density_beta)
    cells_x = rs.uniform(-r_generate, r_generate, num_beta)
    cells_y = rs.uniform(-r_generate, r_generate, num_beta)

    num_amyloids = random.choices(
        [1, 2, 3], weights=[0.8, 0.15, 0.05])[0]
    num_amyloids = num_amyloids if is_t2d else 0
    print('#Amyloids: {}'.format(num_amyloids))

    amyloid_min = amyloid_min if num_amyloids > 1 else amyloid_mu
    amyloid_max = amyloid_max if num_amyloids < 2 else amyloid_mu

    for j in range(num_amyloids):

        r_x, r_y = rs.uniform(amyloid_min, amyloid_max, 2)
        x = islet_radius_x - r_x * 2
        y = islet_radius_y - r_y * 2
        x = rs.uniform(-x, x)
        y = rs.uniform(-y, y)

        islet.amyloids.append([x, y, r_x, r_y])
        print('Amyloid location (x, y): {}, {}; size (r_x, r_y): {} {}'
              .format(x, y, r_x, r_y))

    centers = []
    for i in range(num_beta):

        # Check overlap
        x, y = int(cells_x[i]), int(cells_y[i])
        flag = False
        for (x_, y_) in centers:
            if np.abs(x - x_) < cell_min_dist and np.abs(y - y_) < cell_min_dist:
                flag = True
                break
        if flag:
            continue

        # Remove amyloids
        for (x_, y_, r_x, r_y) in islet.amyloids:
            distance = min(
                r_x - np.abs(x - x_),
                r_y - np.abs(y - y_),
                r_x * r_y - (x - x_) * (x - x_) - (y - y_) * (y - y_)
            )
            if distance > 0:
                flag = True
                break
        if flag:
            continue

        # Set distribution
        distance = max(
            np.abs(x) / islet_radius_x,
            np.abs(y) / islet_radius_y,
            (x*x + y*y) / (islet_radius_x * islet_radius_y)
        )

        if distance < 1:
            weights = (0.05, 0.8, 0, 0.05) if is_t2d else (0.2, 0.8, 0, 0)
        elif distance < thickness_delta:
            weights = (0.2, 0.1, 0.1, 0.6) if is_t2d else (0.8, 0.1, 0.1, 0)
        elif distance < thickness_alpha:
            weights = (0.2, 0.05, 0.05, 0.7) if is_t2d else (0.8, 0.05, 0.05, 0.1)
        else:
            continue

        # Add cell
        cell_type = random.choices(
            ['alpha', 'beta', 'delta', None],
            weights=weights)[0]
        cell_radius = rs.normal(cell_mu, cell_sigma)
        cell_radius = min(cell_radius, cell_max)
        cell_radius = max(cell_radius, cell_min)
        cell_radius = int(cell_radius)
        if cell_type is not None:
            cell = Cell(cell_type, x, y, cell_radius)
            islet.add_cell(cell)
            centers.append([x, y])

    return islet


if __name__ == "__main__":

    for seed in range(5):

        islet = generate_islet(False, seed)
        islet.visualize(save='png')
        islet.save(save='txt')

        islet = generate_islet(True, seed)
        islet.visualize(save='png')
        islet.save(save='txt')
