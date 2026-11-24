# 🎥 Video Input Directory (`inputs/videos/`)

This directory is designated for video recordings that you want to convert into multi-view photo sets for 3D reconstruction.

---

## 📹 Video Capture Best Practices

1. **Keep the Object Stationary**: Walk slowly with your camera around the static object. Do **not** rotate the object on a turntable against a static room background (this violates SfM parallax).
2. **Smooth, Continuous Motion**: Move at a steady pace. Fast camera sweeps cause motion blur which degrades keypoint detection.
3. **Multi-Height Loops**: Record 2 to 3 smooth orbital loops at different elevations (e.g., eye level, overhead 45°, low angle).
4. **Resolution**: 1080p (Full HD) or 4K at 30 or 60 fps. Avoid optical/digital zooming during recording.
5. **Supported Video Formats**: `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`, `.webm`.

---

## ⚡ How to Extract Keyframes from Video

Place your video file inside `inputs/videos/` (e.g. `inputs/videos/my_object.mp4`).

Run the automated keyframe extractor:

```powershell
# Extract sharp, evenly distributed frames directly into inputs/images/my_object
python -m src.extract_frames --video "inputs\videos\my_object.mp4" --output "inputs\images\my_object" --max-frames 60 --sample-fps 2.0
```

### Extractor Features:
- **Sharpness Filtering**: Evaluates Laplacian variance on every candidate frame to discard blurry or out-of-focus frames.
- **Duplicate Suppression**: Compares perceptual thumbnail differences to prevent clustering of redundant, stationary frames.
- **Coverage Distribution**: Evenly spaces selected frames across the entire video duration.
- **JSON Diagnostics**: Generates a `.selection.json` report summarizing decoded, blurry, and selected frame statistics.

---

## 🔄 Next Step: Run 3D Reconstruction

Once frames have been extracted to `inputs/images/my_object/`, run the reconstruction pipeline:

```powershell
# For tabletop masked reconstruction:
python -m src.preprocess_images --images "inputs\images\my_object" --output "prepared\my_object" --max-image-size 1000
python -m src.pipeline --images "prepared\my_object\images" --masks "prepared\my_object\masks" --output "outputs\my_object" --gpu on

# Or for unmasked full scenes:
python -m src.pipeline --images "inputs\images\my_object" --output "outputs\my_object" --gpu on
```
