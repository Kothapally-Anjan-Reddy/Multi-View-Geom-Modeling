"""Image-to-3D pipeline: COLMAP SfM/MVS + Open3D mesh/voxel representations."""
import argparse
import json
import shutil
import subprocess
import sys
import re
import struct
from datetime import datetime, timezone
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def run(command, log_path, cwd=None):
    print("+", " ".join(str(x) for x in command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(str(x) for x in command) + "\n")
        log.flush()
        proc = subprocess.run([str(x) for x in command], cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"Command failed ({proc.returncode}). Inspect {log_path}")

def check_images(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")
    images = sorted(x for x in folder.iterdir() if x.is_file() and x.suffix in IMAGE_SUFFIXES)
    if len(images) < 3:
        raise ValueError(f"Need at least 3 images (20–100+ recommended); found {len(images)} in {folder}")
    return images

def main():
    parser = argparse.ArgumentParser(description="Reconstruct an object and export point cloud, mesh, voxel grid.")
    default_img = Path("inputs/images") if Path("inputs/images").is_dir() else Path("images")
    parser.add_argument("--images", type=Path, default=default_img, help="Image folder; e.g. inputs/images/<dataset>")
    parser.add_argument("--masks", type=Path, help="Foreground mask folder; requires <image-name>.png for every image")
    parser.add_argument("--output", type=Path, help="Output folder; defaults to outputs/<dataset> per run")
    parser.add_argument("--colmap", default="colmap", help="COLMAP executable path")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="Fraction of mesh bounding-box diagonal")
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--gpu", choices=["on", "off"], default="on",
                        help="Use GPU for feature extraction and matching (default: on, for NVIDIA CUDA laptop)")
    parser.add_argument("--gpu-index", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--max-image-size", type=int, default=1600,
                        help="Maximum dense stereo image dimension in pixels; lower if CUDA runs out of memory")
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    parser.add_argument("--sequential-overlap", type=int, default=10, help="Number of neighboring frames to match; sequential matching uses no long-range pairs")
    parser.add_argument("--match-ratio", type=float, default=0.8, help="SIFT match ratio threshold; lower rejects more ambiguous matches")
    parser.add_argument("--skip-dense", action="store_true", help="SfM-only; exports sparse point cloud, no mesh/voxels")
    parser.add_argument("--overwrite", action="store_true", help="Remove previous reconstruction outputs")
    args = parser.parse_args()
    if not 0 < args.voxel_size < 1:
        parser.error("--voxel-size must be a fraction between 0 and 1")
    if args.max_image_size < 256:
        parser.error("--max-image-size must be at least 256")
    if args.sequential_overlap < 2 or not 0 < args.match_ratio < 1:
        parser.error("--sequential-overlap must be >= 2 and --match-ratio must be between 0 and 1")
    images = args.images.resolve()
    image_files = check_images(images)
    masks = args.masks.resolve() if args.masks else None
    if masks:
        from src.masking import validate_masks
        validate_masks(image_files, masks)
    dataset_name = images.parent.name if images.name == 'images' and (images.parent/'preprocessing.json').is_file() else images.name
    if args.output is None:
        clean_name = re.sub(r'[^a-z0-9]+', '_', dataset_name.lower()).strip('_') or 'object'
        stem = Path('outputs') / clean_name
        args.output = stem
        number = 2
        while args.output.exists():
            args.output = Path('outputs') / f'{clean_name}_{number}'
            number += 1
    output = args.output.resolve()
    executable = shutil.which(args.colmap)
    if executable is None:
        raise SystemExit("COLMAP executable not found. Install COLMAP separately and put 'colmap' on PATH, or pass --colmap PATH.")
    if args.overwrite and output.exists():
        # Prevent accidental deletion of the photo source folder.
        if any(output == source or output in source.parents or source in output.parents for source in [images] + ([masks] if masks else [])):
            parser.error("--output must not be the image folder or one of its parents/children when using --overwrite")
        shutil.rmtree(output)
    elif output.exists() and any(output.iterdir()):
        parser.error("Output folder already contains files. Choose a new --output folder to preserve this reconstruction.")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"name": dataset_name.replace('_', ' ').replace('-', ' '),
                "input_images": len(image_files), "source": str(images), "masks": str(masks) if masks else None,
                "created_at": datetime.now(timezone.utc).isoformat()}
    (output/'project.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    from PIL import Image, ImageOps
    with Image.open(image_files[len(image_files)//2]) as source:
        preview = ImageOps.exif_transpose(source).convert('RGB')
        preview.thumbnail((800, 600))
        preview.save(output/'preview.jpg', quality=85)
    log = output / "colmap.log"
    db = output / "database.db"
    sparse = output / "sparse"
    dense = output / "dense"
    sparse.mkdir(exist_ok=True)
    dense.mkdir(exist_ok=True)
    print(f"COLMAP feature GPU acceleration: {args.gpu}; dense PatchMatch requires a compatible GPU build.", flush=True)
    run([executable, "feature_extractor", "--database_path", db, "--image_path", images,
         "--ImageReader.single_camera", "1", "--ImageReader.camera_model", args.camera_model,
         "--FeatureExtraction.use_gpu", "1" if args.gpu == "on" else "0"] + (["--ImageReader.mask_path", masks] if masks else []), log)
    matcher = "exhaustive_matcher" if args.matcher == "exhaustive" else "sequential_matcher"
    run([executable, matcher, "--database_path", db, "--FeatureMatching.use_gpu", "1" if args.gpu == "on" else "0",
         "--SiftMatching.max_ratio", str(args.match_ratio)] +
        (["--SequentialMatching.overlap", str(args.sequential_overlap),
          "--SequentialMatching.quadratic_overlap", "0", "--SequentialMatching.loop_detection", "0"]
         if args.matcher == "sequential" else []), log)
    run([executable, "mapper", "--database_path", db, "--image_path", images, "--output_path", sparse], log)
    def registered_images(path):
        with (path / "images.bin").open("rb") as stream:
            return struct.unpack("<Q", stream.read(8))[0]
    models = sorted((p for p in sparse.iterdir() if p.is_dir() and (p/"images.bin").exists()),
                    key=registered_images, reverse=True)
    if not models:
        raise RuntimeError("No SfM model registered. Check image overlap, sharpness, parallax, and log.")
    model = models[0]
    print(f"Selected sparse model {model.name}: {registered_images(model)}/{len(image_files)} registered images", flush=True)
    if registered_images(model) < max(3, len(image_files) // 2):
        raise RuntimeError("Insufficient registered images for a reliable reconstruction; dense processing was not started.")
    run([executable, "model_converter", "--input_path", model, "--output_path", output/"sparse.ply",
         "--output_type", "PLY"], log)
    if args.skip_dense:
        print("Sparse reconstruction completed:", output/"sparse.ply")
        return
    run([executable, "image_undistorter", "--image_path", images, "--input_path", model,
         "--output_path", dense, "--output_type", "COLMAP"], log)
    run([executable, "patch_match_stereo", "--workspace_path", dense, "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "true",
         "--PatchMatchStereo.gpu_index", str(args.gpu_index),
         "--PatchMatchStereo.max_image_size", str(args.max_image_size)], log)
    if masks:
        from src.masking import stage_mask_images, filter_depth_maps
        mask_input = output/'mask_input'
        mask_workspace = output/'mask_workspace'
        stage_mask_images(image_files, masks, mask_input)
        # Identical model and undistorter options give identical pixel geometry.
        run([executable, "image_undistorter", "--image_path", mask_input,
             "--input_path", model, "--output_path", mask_workspace,
             "--output_type", "COLMAP"], log)
        mask_stats = filter_depth_maps(dense, mask_workspace)
        (output/'masking.json').write_text(json.dumps(mask_stats, indent=2), encoding='utf-8')
        with log.open('a', encoding='utf-8') as stream:
            stream.write("\nForeground depth filtering: " + json.dumps(mask_stats) + "\n")
        print('Foreground depth filtering:', mask_stats, flush=True)
    fused = output/"point_cloud.ply"
    run([executable, "stereo_fusion", "--workspace_path", dense, "--workspace_format", "COLMAP",
         "--input_type", "geometric", "--output_path", fused], log)
    mesh = output/"mesh.ply"
    run([executable, "poisson_mesher", "--input_path", fused, "--output_path", mesh], log)
    # Import Open3D only after external reconstruction has succeeded.
    from src.representations import export_representations
    stats = export_representations(fused, mesh, output, args.voxel_size)
    stats["input_images"] = len(image_files)
    stats["sparse_model"] = str(model)
    (output/"metrics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("Completed. Outputs:", fused, mesh, output/"voxels.ply", output/"comparison.png", sep="\n")

if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
