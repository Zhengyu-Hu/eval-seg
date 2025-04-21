import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image
import shutil

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10") # tab10 is a good color map for visualization
        cmap_idx = 0 if obj_id is None else obj_id # choose a color from tab10 based on the object id
        color = np.array([*cmap(cmap_idx)[:3], 0.6]) # 0.6 is the alpha value
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def png2jpeg(png_file, save_dir):
    img = Image.open(png_file)
    img = img.convert('RGB')
    os.makedirs(save_dir, exist_ok=True)
    img.save(os.path.join(save_dir, os.path.basename(png_file).replace('.png', '.jpeg')), quality=100)

def batch_png2jpeg(pngs_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for png_file in os.listdir(pngs_dir):
        if png_file.endswith('.png'):
            png2jpeg(os.path.join(pngs_dir, png_file), save_dir)

if __name__ == '__main__':
    # Convert png to jpeg
    exp_name = 'fern'
    pngs_dir = 'render_result/raw/'+exp_name
    save_dir = 'render_result/'+exp_name
    batch_png2jpeg(pngs_dir, save_dir)


   