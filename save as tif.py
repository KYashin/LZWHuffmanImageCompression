import cv2
import numpy as np

# # Примеры использования:
# generate_gray_image(64, 64, 0, "black")  # полностью чёрное
# generate_gray_image(64, 64, 255, "white")  # полностью белое

# def generate_one_color_image(height, width, color):
#     # Создание изображения
#     image = np.full((height, width, 3), color, dtype=np.uint8)
#
#     # cv2.imwrite(f'Images/blue_{width}x{height}.tif', image)
#     # cv2.imwrite(f'Images/blue_{width}x{height}.jpg', image)
#     cv2.imwrite(f'Images/blue_{width}x{height}.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 20])
#
# generate_one_color_image(64, 64, (255, 0, 0))


image = cv2.imread('Images/cup.tif')

def add_gaussian_noise(image, mean=0, std=10):
    noise = np.random.normal(mean, std, image.shape).astype(np.int16)
    noisy = image.astype(np.int16) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_salt_and_pepper_noise(image, amount=0.01, salt_vs_pepper=0.5):
    noisy = image.copy()
    num_pixels = int(amount * image.size)

    # Salt (белые точки)
    coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = [255, 255, 255]  # Цвет соли (белый)

    # Pepper (черные точки)
    coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = [0, 0, 0]  # Цвет перца (черный)

    return noisy

def add_poisson_noise(image):
    noisy = np.random.poisson(image.astype(np.uint8)).astype(np.uint8)
    return noisy

def add_speckle_noise(image):
    noise = np.random.randn(*image.shape)  # стандартное нормальное распределение
    noisy = image + image * noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

noisy_gaussian = add_gaussian_noise(image)
noisy_snp = add_salt_and_pepper_noise(image)
noisy_poisson = add_poisson_noise(image)
noisy_speckle = add_speckle_noise(image)

cv2.imwrite('Images/noisy_gaussian.tif', noisy_gaussian)
cv2.imwrite('Images/noisy_snp.tif', noisy_snp)
cv2.imwrite('Images/noisy_poisson.tif', noisy_poisson)
cv2.imwrite('Images/noisy_speckle.tif', noisy_speckle)