
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
         