"""Exports Open3D point cloud, triangle mesh and voxel grid, plus preview figures."""
import json
import math
from pathlib import Path
import numpy as np

def export_representations(point_path: Path, mesh_path: Path, output: Path, voxel_fraction: float):
    import open3d as o3d
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    pcd = o3d.io.read_point_cloud(str(point_path))
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not pcd.has_points():
        raise RuntimeError("Dense point cloud is empty.")
    if len(mesh.triangles) == 0:
        raise RuntimeError("Mesh is empty; inspect dense reconstruction and try a better image set.")
    mesh.compute_vertex_normals()
    bbox = mesh.get_axis_aligned_bounding_box()
    diagonal = float(np.linalg.norm(bbox.get_extent()))
    if diagonal <= 0:
        raise RuntimeError("Degenerate mesh bounding box.")
    voxel_size = diagonal * voxel_fraction
    grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
    if len(grid.get_voxels()) == 0:
        raise RuntimeError("Voxelization returned zero voxels.")
    # Open3D does not serialize VoxelGrid directly to PLY: export occupied voxel centers as a PLY point cloud,
    # and preserve exact integer grid coordinates and scale in voxels.npz for lossless reloading.
    voxels = grid.get_voxels()
    ijk = np.array([v.grid_index for v in voxels], dtype=np.int32)
    origin = np.asarray(grid.origin, dtype=float)
    centers = origin + (ijk.astype(float) + 0.5) * voxel_size
    vp = o3d.geometry.PointCloud()
    vp.points = o3d.utility.Vector3dVector(centers)
    o3d.io.write_point_cloud(str(output/"voxels.ply"), vp)
    np.savez_compressed(output/"voxels.npz", indices=ijk, origin=origin, voxel_size=voxel_size)
    stats = {
        "dense_points": len(pcd.points), "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles), "occupied_voxels": len(voxels),
        "voxel_size_scene_units": voxel_size,
        "note": "SfM scale is arbitrary without an external scale reference; voxel size uses mesh diagonal."
    }
    # Matplotlib plots: subsample for manageable report preview.
    fig = plt.figure(figsize=(14, 4.5))
    ax = fig.add_subplot(131, projection="3d")
    pts = np.asarray(pcd.points)
    sample = pts[::max(1, len(pts)//7000)]
    ax.scatter(sample[:,0], sample[:,1], sample[:,2], s=.2)
    ax.set_title(f"Point cloud ({len(pts):,} points)")
    ax = fig.add_subplot(132, projection="3d")
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    tris_sample = tris[::max(1,len(tris)//1800)]
    poly = Poly3DCollection(verts[tris_sample], alpha=.8, linewidths=.05)
    ax.add_collection3d(poly)
    ax.auto_scale_xyz(verts[:,0], verts[:,1], verts[:,2])
    ax.set_title(f"Mesh ({len(tris):,} triangles)")
    ax = fig.add_subplot(133, projection="3d")
    subset = centers[::max(1,len(centers)//6000)]
    ax.scatter(subset[:,0], subset[:,1], subset[:,2], s=2, marker="s")
    ax.set_title(f"Voxel centers ({len(centers):,} voxels)")
    for a in fig.axes:
        a.set_xlabel("X"); a.set_ylabel("Y"); a.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(output/"comparison.png", dpi=170)
    plt.close(fig)
    return stats
