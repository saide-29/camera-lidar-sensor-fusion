import cv2


    
  

image_path = "data/kitti/image_2/image_2/000000.png"

image = cv2.imread(image_path)

print("Image loaded:", image is not None)

if image is not None:
    print("Shape:", image.shape)




cv2.imshow("KITTI IMAGE " , image)

while True:
    key = cv2.waitKey(1)
    if key == 27: 
        break

cv2.destroyAllWindows  