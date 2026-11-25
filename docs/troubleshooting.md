# Troubleshooting & Technical FAQ

This document covers common setup issues, CUDA memory constraints, COLMAP binary configurations, and Git LFS troubleshooting.

---

## 🔧 1. COLMAP Installation & Path Setup

### Windows
1. Download pre-compiled binary with CUDA support: [COLMAP GitHub Releases](https://github.com/colmap/colmap/releases) (e.g. `COLMAP-3.8-windows-cuda.zip` or newer).
2. Extract the archive to a clean folder (e.g., `D:\tools\COLMAP` or `C:\Program Files\COLMAP`).
3. Verify by running in PowerShell:
   ```powershell
   & "D:\tools\COLMAP\bin\colmap.exe" -h
   ```
4. If you have extracted COLMAP to a custom directory, pass it via CLI:
   ```powershell
   python -m src.pipeline --colmap "D:\tools\COLMAP\bin\colmap.exe" ...
   ```
   Or add `D:\tools\COLMAP\bin` to your system `PATH` environment variable.

### Linux (Ubuntu 20.04 / 22.04)
```bash
sudo apt update
sudo apt install colmap
colmap -h
```

---

## ⚡ 2. GPU & CUDA Memory Management (Out of Memory)

If COLMAP crashes during **PatchMatch Stereo** with a CUDA Out of Memory (OOM) error:
```
CUDA error: out of memory
```

### Solutions:
1. **Reduce Maximum Image Dimension**:
   Lower `--max-image-size` to `1000` or `800` pixels:
   ```powershell
   python -m src.pipeline --max-image-size 1000 ...
   ```
2. **Close Background GPU Applications**:
   Close web browsers, video players, or 3D games consuming VRAM before running dense reconstruction.
3. **Use Google Colab Cloud GPU**:
   Use our pre-configured [`3D_Reconstruction_Colab.ipynb`](../3D_Reconstruction_Colab.ipynb) notebook on Google Colab with 15 GB of Free T4 GPU VRAM.

---

## 📦 3. Git LFS & GitHub 100MB File Size Limit

### Symptom:
When executing `git push`, GitHub rejects your commit with:
```
remote: error: GH001: Large files detected.
remote: error: File outputs/Building/mesh.ply is 194.10 MB; this exceeds GitHub's file size limit of 100.00 MB
```

### Solution:
Install and initialize **Git Large File Storage (Git LFS)** before adding files:

```bash
# 1. Download Git LFS from https://git-lfs.com or run:
git lfs install

# 2. Verify .gitattributes is tracked:
git add .gitattributes

# 3. Add files and commit:
git add .
git commit -m "feat: commit project with git lfs"
git push origin main
```

The repository's [`.gitattributes`](../.gitattributes) is already configured to automatically track:
- `*.ply`
- `*.npz`
- `*.mp4`
- `*.pdf`

---

## 🌐 4. Interactive WebGL Three.js Viewer FAQ

### Q: The viewer page loads, but models do not appear in the dropdown.
- **Cause**: The viewer defaults to scanning the `outputs/` directory.
- **Fix**: Ensure your models are stored inside `outputs/<dataset_name>/` containing `point_cloud.ply` and `mesh.ply`. Click **Refresh outputs** in the browser header.

### Q: Port 8000 is already in use.
- **Fix**: Specify an alternate port:
  ```powershell
  python -m src.viewer --port 8080
  ```
  Then open `http://localhost:8080` in your browser.

---

## 🎭 5. Neural Segmentation (`rembg`) Offline Usage

The `rembg` library automatically downloads the `isnet-general-use.onnx` model (approx. 175 MB) to `~/.u2net/` on its first run.
- If running on an offline machine, pre-download `isnet-general-use.onnx` and place it into `%USERPROFILE%\.u2net\isnet-general-use.onnx` (or `~/.u2net/isnet-general-use.onnx` on Linux/macOS).
- Ensure input images are high-resolution; `src.preprocess_images` normalizes EXIF orientation automatically.
