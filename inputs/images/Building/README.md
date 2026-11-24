Place 20–100+ overlapping JPG/PNG photos of one stationary, textured object here. Keep background consistent and move the camera around the object; do not rotate a glossy object against a fixed background.

For foreground-only reconstruction, follow the RGB + separate mask workflow in
../../README.md. Prepare originals into a new folder and inspect segmentation:
automatic foreground detection may not select an entire building correctly.
Do not paint backgrounds blue. Masks exclude background features and geometric
depths before fusion, but cannot recover unobserved building surfaces.
