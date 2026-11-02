import numpy as np
import cv2
from load_lidar import load_lidar
from load_calib import load_calibration

image_path = "data/kitti/image_2/image_2/000000.png"

image = cv2.imread(image_path)

print(image.shape)



lidar_path = "data/kitti/velodyne/velodyne/000000.bin"

points = load_lidar(lidar_path)



calib_path = "data/kitti/calib/calib/000000.txt"

P2, R0_rect, Tr_velo_to_cam = load_calibration(calib_path)

def project_points(points, P2, R0_rect, Tr_velo_to_cam):

    xyz = points[:, :3]
    N = xyz.shape[0] # all points
    point_h = np.hstack([xyz, np.ones((N, 1))]) 

    R0_rect_h = np.eye(4) # I @ A = A 
    R0_rect_h[:3,:3] = R0_rect

    Tr_velo_to_cam_h = np.eye(4)
    Tr_velo_to_cam_h[:3, :] = Tr_velo_to_cam

    camera_view = (Tr_velo_to_cam_h @ point_h.T).T
    
    front_mask = camera_view[:, 2] > 0
    camera_view = camera_view[front_mask]

    camera_rect_h = (R0_rect_h @ camera_view.T).T

    pixel_view = (P2 @ camera_rect_h.T).T
    u = pixel_view[:, 0] / pixel_view[:, 2]#Perspective Projection
    v = pixel_view[:, 1] / pixel_view[:, 2]
    depth = camera_rect_h[:, 2] # camera depth (Z)
 
    return u, v, depth

print("LiDAR:", points.shape)
print("P2:", P2.shape)
print("Tr_velo_to_cam:", Tr_velo_to_cam.shape)

u, v, depth = project_points(
    points,
    P2,
    R0_rect,
    Tr_velo_to_cam
)




img_mask = (u >= 0) & (u < image.shape[1]) & (v >= 0) & (v < image.shape[0])
u = u[img_mask].astype(int)
v = v[img_mask].astype(int)
depth = depth[img_mask]

depth_clipped = np.clip(depth, 0, 40)
norm = (depth_clipped / 40 * 255).astype(np.uint8)
colors = cv2.applyColorMap(norm.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)

for (uu, vv), color in zip(zip(u, v), colors):
    cv2.circle(image, (uu, vv), 1, tuple(int(c) for c in color), -1)
   
print(points.shape)
print(points[0])
print(points[1])
print(points[2])

cv2.imshow("Sensor Fusion", image)

cv2.waitKey(0)

cv2.destroyAllWindows()


