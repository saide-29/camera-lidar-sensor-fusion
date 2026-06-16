import numpy as np 
import open3d as o3d



lidar_path = "data/kitti/velodyne/velodyne/000000.bin"

points = np.fromfile(
    lidar_path,
    dtype=np.float32
)

points = points.reshape(-1,4)
xyz = points[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)


o3d.visualization.draw_geometries([pcd])
