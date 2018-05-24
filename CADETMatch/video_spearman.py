import sys
from cache import cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy

def main():
    cache.setup(sys.argv[1])
    cache.progress_path = Path(cache.settings['resultsDirBase']) / "progress.csv"

    output_spear = cache.settings['resultsDirSpace'] / "spearman"

    pngs = list(output_spear.glob('*.png'))

    pngs_sorted = sorted( pngs, key = lambda file: file.stat().st_ctime)

    fig = plt.figure(figsize=(10,10), dpi=100)

    ims = []
    for png in pngs_sorted:
        with png.open('rb') as image_file:
            data = plt.imread(image_file)
        ymax, xmax, _ = data.shape
        im = plt.imshow(data, animated=True, aspect="equal", extent=[0,xmax,ymax,0])
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=False,
                                repeat=False)

    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=5, bitrate=50000)
    fig.tight_layout()

    ani.save(str(output_spear / 'spearman.mp4'), writer=writer)

if __name__ == "__main__":
    main()

