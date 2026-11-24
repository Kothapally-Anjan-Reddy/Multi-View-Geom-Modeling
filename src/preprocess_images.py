"""Prepare resized images on white backgrounds and separate foreground masks."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import os

from PIL import Image, ImageOps, ImageFilter

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def transform(image, segment, max_size, threshold=240, erode=2):
    image = ImageOps.exif_transpose(image).convert('RGB')
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    cutout = segment(image).convert('RGBA')
    if cutout.size != image.size:
        raise ValueError('Segmentation returned an unexpected image size.')
    mask = cutout.getchannel('A').point(lambda a: 255 if a >= threshold else 0)
    if erode:
        mask = mask.filter(ImageFilter.MinFilter(2*erode+1))
    if mask.getbbox() is None:
        raise ValueError('No foreground detected. Review the segmentation model/settings.')
    return image, mask


def preprocess(folder, max_size=1000, model='isnet-general-use', segment=None,
               output=None, threshold=240, erode=2, background='white'):
    folder = Path(folder).resolve()
    output = Path(output or Path('prepared')/folder.name).resolve()
    if not folder.is_dir():
        raise ValueError(f'Image folder not found: {folder}')
    if max_size < 64 or not 1 <= threshold <= 255 or not 0 <= erode <= 20:
        raise ValueError('Require max-image-size >= 64, threshold 1..255, erode 0..20.')
    if background not in ('white', 'original'):
        raise ValueError('Background must be white or original.')
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Choose a new --output folder.')
    if output == folder or folder in output.parents or output in folder.parents:
        raise ValueError('Prepared output must be separate from the source image folder.')
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise ValueError(f'No supported images in {folder}')
    for path in files:
        with Image.open(path) as image:
            image.verify()
    if segment is None:
        try:
            from rembg import new_session, remove
        except ImportError as exc:
            raise RuntimeError('Install requirements-preprocess.txt before preprocessing.') from exc
        print(f'Loading {model} on CPU (first use downloads weights)...', flush=True)
        session = new_session(model, providers=['CPUExecutionProvider'])
        segment = lambda image: remove(image, session=session)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare_', dir=output.parent) as temp:
        staged = Path(temp)/'dataset'
        (staged/'images').mkdir(parents=True)
        (staged/'masks').mkdir()
        for index, path in enumerate(files, 1):
            with Image.open(path) as image:
                rgb, mask = transform(image, segment, max_size, threshold, erode)
            if background == 'white':
                rgb = Image.composite(rgb, Image.new('RGB', rgb.size, 'white'), mask)
            # Append rather than replace suffix to avoid collisions (a.jpg vs a.png).
            name = path.name+'.png'
            rgb.save(staged/'images'/name)
            mask.save(staged/'masks'/(name+'.png'))
            print(f'[{index}/{len(files)}] {name}: {rgb.width} x {rgb.height}', flush=True)
        (staged/'preprocessing.json').write_text(json.dumps({
            'source_folder':str(folder), 'max_image_size':max_size, 'model':model,
            'threshold':threshold, 'erode_pixels':erode, 'files':[p.name for p in files],
            'background':background, 'mask_convention':'white=foreground, black=excluded',
        }, indent=2), encoding='utf-8')
        os.rename(staged, output)
    print(f'Prepared images: {output / "images"}\nForeground masks: {output / "masks"}\nOriginals unchanged.', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True, help='Original photo folder (use backup if previously painted blue)')
    parser.add_argument('--output', type=Path, help='New dataset folder (default: prepared/<source-folder-name>)')
    parser.add_argument('--max-image-size', type=int, default=1000)
    parser.add_argument('--model', choices=['isnet-general-use', 'u2net', 'u2netp'], default='isnet-general-use')
    parser.add_argument('--threshold', type=int, default=240, help='Minimum foreground alpha, 1..255')
    parser.add_argument('--erode', type=int, default=2, help='Inset mask boundary in resized-image pixels, 0..20')
    parser.add_argument('--background', choices=['white', 'original'], default='white')
    args = parser.parse_args()
    preprocess(args.images, args.max_image_size, args.model, output=args.output,
               threshold=args.threshold, erode=args.erode, background=args.background)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
