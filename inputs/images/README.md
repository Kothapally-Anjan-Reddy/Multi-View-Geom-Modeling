# 📸 Photographic Input Datasets (`inputs/images/`)

This directory contains multi-view photo sets for 3D reconstruction. Each subfolder represents an individual subject or scene.

---

## 📷 Photography Guidelines for High-Quality 3D Reconstruction

To ensure 100% camera registration and dense geometric recovery:
1. **Overlap**: Maintain **60% to 80% visual overlap** between consecutive photographs.
2. **Trajectory**: Walk or orbit around the object in a 360° circle at 1 to 3 different elevation angles (e.g., eye level, 45° overhead, low angle).
3. **Lighting**: Use diffuse, uniform illumination. Avoid harsh dynamic shadows, direct sunlight flares, or reflective mirrors.
4. **Surface Texture**: The object should have visible surface texture (e.g., stone, wood, cloth, matte plastic). Pure featureless white walls or specular mirrors will degrade SfM matching.
5. **Quantity**: 
   - Small tabletop object: 30–75 photos.
   - Large monument or room: 80–150+ photos.

---

## 📁 How to Add a New Dataset

Create a new subfolder under `inputs/images/`:
```
inputs/images/
└── my_custom_object/
    ├── img_0001.jpg
    ├── img_0002.jpg
    ├── img_0003.jpg
    └── ...
```

---

## ⚡ Execution Commands

### Case A: Tabletop / Turntable Object (With Neural Background Masking)
If your subject sits on a table or background that should be removed:

```powershell
# 1. Automated neural foreground segmentation (rembg)
python -m src.preprocess_images --images "inputs\images\my_custom_object" --output "prepared\my_custom_object" --max-image-size 1000

# 2. End-to-end Multi-View 3D Reconstruction
python -m src.pipeline --images "prepared\my_custom_object\images" --masks "prepared\my_custom_object\masks" --output "outputs\my_custom_object" --gpu on --matcher exhaustive --max-image-size 1000
```

### Case B: Architecture, Rooms, or Full Scenes (Without Masking)
To reconstruct the entire environment including the background:

```powershell
python -m src.pipeline --images "inputs\images\my_custom_object" --output "outputs\my_custom_object" --gpu on --matcher exhaustive --max-image-size 1000
```

---

## 🌐 View Reconstructed 3D Models

Start the interactive WebGL Three.js localhost viewer:
```powershell
python -m src.viewer --port 8000
```
Open **`http://localhost:8000`** in your browser. Your newly reconstructed dataset will automatically appear in the model gallery!
