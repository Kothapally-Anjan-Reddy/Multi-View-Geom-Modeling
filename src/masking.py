"""Validate masks, align them with COLMAP undistortion, and gate dense depths."""
from pathlib import Path
import os
import numpy as np
from PIL import Image, ImageFilter


def validate_masks(images, masks):
    for path in images:
        mask_path = masks/(path.name+'.png')
        if not mask_path.is_file():
            raise ValueError(f'Missing foreground mask: {mask_path}')
        with Image.open(path) as image, Image.open(mask_path) as mask:
            if image.getexif().get(274, 1) != 1:
                raise ValueError('Normalize EXIF orientation with preprocess_images before using masks.')
            if image.size != mask.size:
                raise ValueError(f'Mask/image dimensions differ: {path.name}')
            values = np.asarray(mask.convert('L'))
            if not np.isin(values, [0, 255]).all() or not np.any(values):
                raise ValueError(f'Mask must be binary with nonempty foreground: {mask_path}')


def stage_mask_images(images, masks, target):
    target.mkdir(parents=True)
    for path in images:
        with Image.open(masks/(path.name+'.png')) as mask:
            options = {'quality':100, 'subsampling':0} if path.suffix.lower() in {'.jpg','.jpeg'} else {}
            mask.convert('RGB').save(target/path.name, **options)


def filter_depth_maps(dense, mask_workspace):
    paths = sorted((dense/'stereo'/'depth_maps').glob('*.geometric.bin'))
    if not paths:
        raise RuntimeError('No geometric depth maps available to mask.')
    removed = 0
    for path in paths:
        name = path.name.removesuffix('.geometric.bin')
        with path.open('rb') as stream:
            header = bytearray()
            while header.count(b'&') < 3:
                char = stream.read(1)
                if not char or len(header) > 100:
                    raise ValueError(f'Invalid COLMAP depth header: {path}')
                header.extend(char)
            width, height, channels = map(int, header[:-1].split(b'&'))
            if channels != 1 or width <= 0 or height <= 0:
                raise ValueError(f'Invalid depth dimensions: {path}')
            depth = np.fromfile(stream, dtype='<f4')
        if depth.size != width*height:
            raise ValueError(f'Truncated depth map: {path}')
        depth = depth.reshape(height, width)
        with Image.open(mask_workspace/'images'/name) as mask, Image.open(dense/'images'/name) as rgb:
            if mask.size != rgb.size:
                raise ValueError(f'Undistorted mask/image dimensions differ: {name}')
            # Conservative area downsampling rejects mixed boundary pixels.
            mask = mask.convert('L').resize((width,height), Image.Resampling.BOX)
            mask = mask.point(lambda a: 255 if a >= 254 else 0).filter(ImageFilter.MinFilter(3))
            keep = np.asarray(mask) == 255
        removed += int(np.count_nonzero((depth > 0) & ~keep))
        depth[~keep] = 0
        temp = path.with_suffix('.masking.tmp')
        with temp.open('wb') as stream:
            stream.write(header)
            depth.astype('<f4').tofile(stream)
        os.replace(temp, path)
    return {'views':len(paths), 'background_depth_samples_removed':removed}
