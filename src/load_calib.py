
import numpy as np 

calib_path = "data/kitti/calib/calib/000000.txt"

with open(calib_path, "r") as f:
      lines = f.readlines()


for line in lines:
      if line.startswith("P2:"):
          
          parts = line.split()
          values = np.array(parts[1:], dtype=float)
          
          P2 = values.reshape(3,4)

          print(P2)
          print(P2.shape)
         
      if line.startswith("Tr_velo_to_cam:"):
           
           parts_velo = line.split()
           values1 = np.array(parts_velo[1:], dtype=float)

           Tr_velo_to_cam = values1.reshape(3,4)

           print(Tr_velo_to_cam)
           print(Tr_velo_to_cam.shape)

point = np.array([
      18.324,
      0.049,
      0.829
])


point_h = np.append(point, 1) #lidar point

print(point_h)
print(point_h.shape)


camera_view = Tr_velo_to_cam @ point_h #Lidar points -> camera points
print(camera_view)
print(camera_view.shape)


camera_view_h = np.append(camera_view, 1)

print(camera_view_h)
print(camera_view_h.shape)
       
pixel_view = P2 @ camera_view_h # camera points -> pixel (a,b,c)

print(pixel_view)
print(pixel_view.shape)


##### Image 2D #####

u = pixel_view[0] / pixel_view[2] #normalization a/c b/c
v = pixel_view[1] / pixel_view[2]

print("u =", u)
print("v =", v)
