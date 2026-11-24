# 📥 Unified Project Inputs Directory

Welcome to the central inputs directory for the **Multi-View 3D Reconstruction Pipeline**. All photographic and video input assets are organized in this folder.

---

## 📁 Directory Structure

```
inputs/
├── images/                        # Multi-view photograph datasets (folder-per-dataset)
│   ├── README.md                  # Comprehensive photography & reconstruction guide
│   ├── owl-images/                # Stone owl statue (30 photos)
│   ├── Building/                  # South Building architecture (128 photos)
│   ├── Dog_RGB/                   # Dog organic figurine (72 photos)
│   ├── Apple/                     # Glossy fruit specular baseline (72 photos)
│   ├── Cat_RGB/                   # Cat figurine dataset (108 photos)
│   └── <your_custom_dataset>/     # Place your custom image subfolder here
│
└── videos/                        # Video recordings for automated frame extraction
    └── README.md                  # Video capture guidelines & extraction commands
```

---

## 🚀 Quick Navigation & Workflow

### 1. Reconstructing from Photographs (`inputs/images/`)
- Place a subfolder of 30–100+ overlapping photographs inside [`inputs/images/<your_dataset_name>/`](images/).
- See detailed instructions and commands in [**`inputs/images/README.md`**](images/README.md).

```powershell
# Example: Run 3D reconstruction on a custom image dataset
python -m src.pipeline --images "inputs\images\my_object" --output "outputs\my_object" --gpu on
```

### 2. Reconstructing from Video (`inputs/videos/`)
- Place a video file (`.mp4`, `.mov`, `.avi`, `.mkv`) inside [`inputs/videos/`](videos/).
- Automatically extract sharp, non-blurry, evenly-distributed keyframes into `inputs/images/`:
- See detailed instructions in [**`inputs/videos/README.md`**](videos/README.md).

```powershell
# Example: Extract keyframes from video into inputs/images/my_video_frames
python -m src.extract_frames --video "inputs\videos\my_recording.mp4" --output "inputs\images\my_video_frames"
```

Once frames are extracted, feed `inputs\images\my_video_frames` directly into `src.pipeline`!
