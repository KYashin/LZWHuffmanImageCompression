import cv2
import numpy as np

img_jpg = cv2.imread('Images/green_10000x10000.jpg', cv2.IMREAD_COLOR)
img_tif = cv2.imread('Images/green_10000x10000.tif', cv2.IMREAD_COLOR)

diff = cv2.absdiff(img_jpg, img_tif)
if np.all(diff == 0):
    print("diff полностью из нулей.")
else:
    print("Есть отличия.")

# cv2.imshow('Difference', diff)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# def generate_one_color_image(height, width, color):
#     # Создание изображения
#     image = np.full((height, width, 3), color, dtype=np.uint8)
#
#     cv2.imwrite(f'Images/green_{width}x{height}.tif', image)
#     cv2.imwrite(f'Images/green_{width}x{height}.jpg', image)
#
# generate_one_color_image(10000, 10000, (0, 255, 0))