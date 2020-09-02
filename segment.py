import argparse
import matplotlib.pyplot as plt
from skimage.io import imread
from cellpose import models, io, plot


def main(model_type):
    model = models.Cellpose(model_type=model_type)
    # files = ['KRT', 'MCAM', 'SST', 'PPY']
    files = ['KRT']
    imgs = [imread('data/AFFN281-ND-52yM-T1A_{}.tif'.format(f)) for f in files]
    nimg = len(imgs)
    channels = [[0, 0]]
    masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=channels)
    io.masks_flows_to_seg(imgs, masks, flows, diams, files, channels)
    io.save_to_png(imgs, masks, flows, files)

    for idx in range(nimg):
        maski = masks[idx]
        flowi = flows[idx][0]
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, imgs[idx], maski, flowi, channels=channels[idx])
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='cyto',
        help='cyto or nuclei'
    )
    args, _ = parser.parse_known_args()
    main(args.model)
