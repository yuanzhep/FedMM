# 0306_2023,yz

import json
from multiprocessing import Process, JoinableQueue
import argparse
import os
import re
import shutil
import sys
import glob
import numpy as np
import math
from unicodedata import normalize
from skimage import io
from skimage.color import rgb2hsv
from skimage.util import img_as_ubyte
from skimage import filters
from PIL import Image, ImageFilter, ImageStat
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
Image.MAX_IMAGE_PIXELS = None
VIEWER_SLIDE_NAME = 'slide'

#  Generates and writes tiles
class TileWorker(Process):
    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,
                 quality, threshold):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._threshold = threshold
        self._slide = None

    def run(self):
        self._slide = open_slide(self._slidepath)
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data
            try:
                tile = dz.get_tile(level, address)
                edge = tile.filter(ImageFilter.FIND_EDGES)
                edge = ImageStat.Stat(edge).sum
                edge = np.mean(edge) / (self._tile_size ** 2)
                w, h = tile.size
                if edge > self._threshold:
                    if not (w == self._tile_size and h == self._tile_size):
                        tile = tile.resize((self._tile_size, self._tile_size))
                    tile.save(outfile, quality=self._quality)
            except:
                pass
            self._queue.task_done()

    def _get_dz(self):
        image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap,
                                 limit_bounds=self._limit_bounds)

class DeepZoomImageTiler(object):
    def __init__(self, dz, basename, target_level, mag_base, format, associated, queue):
        self._dz = dz
        self._basename = basename
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._target_level = target_level
        self._mag_base = int(mag_base)

    def run(self):
        self._write_tiles()

    def _write_tiles(self):
        # magnification 20
        target_level = self._dz.level_count - 1
        tiledir = os.path.join("%s_files" % self._basename, str(20))
        if not os.path.exists(tiledir):
            os.makedirs(tiledir)
        cols, rows = self._dz.level_tiles[target_level]
        for row in range(rows):
            for col in range(cols):
                tilename = os.path.join(tiledir, '%d_%d.%s' % (
                    col, row, self._format))
                if not os.path.exists(tilename):
                    self._queue.put((self._associated, target_level, (col, row),
                                     tilename))
                self._tile_done()

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                self._associated or 'slide', count, total),
                  end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)

class ZoomTiler:
    def __init__(self, slidepath, destpath, target_level, base_mag, format, tile_size, overlap, limit_bounds, quality, workers, threshold):
        self.slide = open_slide(slidepath)
        self.basename = os.path.splitext(os.path.basename(slidepath))[0]
        self.dz = DeepZoomGenerator(self.slide, tile_size, overlap, limit_bounds=limit_bounds)
        self.format = format
        self.queue = JoinableQueue(workers)
        self.workers = [TileWorker(self.queue, slidepath, tile_size, overlap, limit_bounds, quality, threshold) for _ in range(workers)]
        for worker in self.workers:
            worker.start()
        self._run(target_level, base_mag)

    def _run(self, target_level, base_mag):
        tiler = DeepZoomImageTiler(self.dz, self.basename, target_level, base_mag, self.format, None, self.queue)
        tiler.run()
        self.queue.join()

def single_level_patches(img_slide, out_base, ext='jpeg'):
    print('\n Organizing patches')
    img_name = img_slide.split(os.sep)[-1].split('.')[0]
    img_class = img_slide.split(os.sep)[2]
    bag_path = os.path.join(out_base, img_class, img_name)
    os.makedirs(bag_path, exist_ok=True)
    patches = glob.glob(os.path.join('WSI_temp_files', '20', '*.' + ext))
    for i, patch in enumerate(patches):
        patch_name = patch.split(os.sep)[-1]
        shutil.move(patch, os.path.join(bag_path, patch_name))
        sys.stdout.write('\r Patch [%d/%d]' % (i + 1, len(patches)))
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepZoom Static Tiler')
    parser.add_argument('slidepath', help='Path of the WSI slide')
    parser.add_argument('destpath', help='Path to save the output files')
    parser.add_argument('out_base', help='Path to save the patches')
    parser.add_argument('--format', default='jpeg', help='Format of the image - default is jpeg')
    parser.add_argument('--tile-size', default=2048, type=int, help='Size of the square tiles - default is 2048')
    parser.add_argument('--overlap', default=0, type=int, help='Overlap of the tiles - default is 1')
    parser.add_argument('--base-mag', default=20, type=int, help='Magnification level for generating the tiles - default is 20')
    parser.add_argument('--objective', default=40, type=int, help='Objective of the slide scanner - default is 40')
    parser.add_argument('--workers', default=4, type=int, help='Number of worker processes - default is 4')
    parser.add_argument('--quality', default=75, type=int, help='JPEG quality - default is 75')
    parser.add_argument('--background-t', default=220, type=int, help='Background threshold - default is 220')
    args = parser.parse_args()
    all_slides = glob.glob(os.path.join(args.slidepath, '*.svs'))

    for idx, c_slide in enumerate(all_slides):
        print('Processing slide {}/{}'.format(idx + 1, len(all_slides)))
        ZoomTiler(c_slide, 'WSI_temp', [0], args.base_mag, args.objective, args.format, args.tile_size,
                            args.overlap, True, args.quality, args.workers, args.background_t).run()
        single_level_patches(c_slide, args.out_base, ext=args.format)
        shutil.rmtree('WSI_temp_files')

    print('Patch extraction done for {} slides.'.format(len(all_slides)))

