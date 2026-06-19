import numpy as np
import cv2

image_path = "data/kitti/image_2/image_2/000000.png"

image = cv2.imread(image_path)

print(image.shape)

####

lidar_path = "data/kitti/velodyne/velodyne/000000.bin"

points = np.fromfile(
    lidar_path,
    dtype=np.float32
)

points = points.reshape(-1, 4)

#####

calib_path = "data/kitti/calib/calib/000000.txt"

with open(calib_path, "r") as f:
    lines = f.readlines()

for line in lines:

    if line.startswith("P2:"):
        P2 = np.array(
            line.split()[1:],
            dtype=float
        ).reshape(3, 4)

    if line.startswith("Tr_velo_to_cam:"):
        Tr_velo_to_cam = np.array(
            line.split()[1:],
            dtype=float
        ).reshape(3, 4)

print("LiDAR:", points.shape)
print("P2:", P2.shape)
print("Tr_velo_to_cam:", Tr_velo_to_cam.shape)

####

for point in points[::20]:
    
    xyz = point[:3]

    point_h = np.append(xyz, 1)

    camera_view = Tr_velo_to_cam @ point_h

    
    if camera_view[2] <= 0:
        continue

    camera_view_h = np.append(camera_view, 1)

    pixel_view = P2 @ camera_view_h

    u = pixel_view[0] / pixel_view[2]
    v = pixel_view[1] / pixel_view[2]

    u = int(u)  # Opencv (int(u), int(v))
    v = int(v)

    if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
        cv2.circle(image, (u,v),2, (0, 0, 255), -1)

   


cv2.imshow("Sensor Fusion", image)

cv2.waitKey(0)

cv2.destroyAllWindows()


