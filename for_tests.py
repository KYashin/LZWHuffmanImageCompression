import cv2
import numpy as np
# Допустим, у тебя уже есть изображение:
# image.shape = (6, 8, 3)  — 6 строк, 8 столбцов, 3 канала (BGR)
image = cv2.imread('Images/small.tif')  # или свой массив

# Новый размер (например, увеличить до 160×120)
new_width = 224
new_height = 168

# Увеличение с интерполяцией
resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

# Сохранить или показать
cv2.imwrite(f'Images/small_{new_width}x{new_height}.tif', resized)