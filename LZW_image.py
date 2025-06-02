import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import struct

MAX_DICT_SIZE = 4096

# Упаковка 12-битных кодов
def pack_12bit_codes(codes):
    packed_bytes = bytearray()
    i = 0
    while i < len(codes):
        if i + 1 < len(codes):
            c1, c2 = codes[i], codes[i + 1]
            b1 = (c1 >> 4) & 0xFF
            b2 = ((c1 & 0xF) << 4) | ((c2 >> 8) & 0xF)
            b3 = c2 & 0xFF
            packed_bytes.extend([b1, b2, b3])
            i += 2
        else:
            c1 = codes[i]
            b1 = (c1 >> 4) & 0xFF
            b2 = (c1 & 0xF) << 4
            packed_bytes.extend([b1, b2])
            i += 1
    return bytes(packed_bytes)


# Распаковка 12-битных кодов
def unpack_12bit_codes(data_bytes, expected_count):
    codes = []
    i = 0
    length = len(data_bytes)

    while len(codes) < expected_count:
        if i + 2 < length and len(codes) + 2 <= expected_count:
            b1 = data_bytes[i]
            b2 = data_bytes[i + 1]
            b3 = data_bytes[i + 2]

            code1 = (b1 << 4) | (b2 >> 4)
            code2 = ((b2 & 0x0F) << 8) | b3
            codes.extend([code1, code2])
            i += 3

        elif i + 1 < length and len(codes) < expected_count:
            b1 = data_bytes[i]
            b2 = data_bytes[i + 1]
            code1 = (b1 << 4) | (b2 >> 4)
            codes.append(code1)
            i += 2

        elif i < length and len(codes) < expected_count:
            b1 = data_bytes[i]
            code1 = (b1 << 4)
            codes.append(code1)
            i += 1
        else:
            break

    return codes[:expected_count]  # Добавлено усечение списка


# LZW-компрессия
def lzw_compress(data):
    dictionary = {'0': 0, '1': 1}
    dict_size = 2
    w = ''
    compressed = []

    for c in data:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            compressed.append(dictionary[w])
            if dict_size < MAX_DICT_SIZE:
                dictionary[wc] = dict_size
                dict_size += 1
            w = c

    if w:
        compressed.append(dictionary[w])

    return compressed


# LZW-декомпрессия
def lzw_decompress(compressed):
    dictionary = {0: '0', 1: '1'}
    dict_size = 2
    result = []

    w = dictionary[compressed[0]]
    result.append(w)

    for k in compressed[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError(f"Некорректный сжатый индекс: {k}")

        result.append(entry)
        if dict_size < MAX_DICT_SIZE:
            dictionary[dict_size] = w + entry[0]
            dict_size += 1
        w = entry

    return ''.join(result)


# Преобразование строки в изображение
def bit_string_to_image(bit_string, shape):
    flat = np.array([int(b) for b in bit_string[:shape[0]*shape[1]]], dtype=np.uint8)
    return flat.reshape(shape)

# Основной код
img = cv2.imread(r'Images/apple.png', cv2.IMREAD_GRAYSCALE)

bit_planes = []
for i in range(8):
    bit_plane = ((img >> i) & 1).astype(np.uint8)
    bit_planes.append(bit_plane)

bit_plane_strings = []
for plane in bit_planes:
    flat = plane.flatten()
    bit_string = ''.join(str(b) for b in flat)
    bit_plane_strings.append(bit_string)

original_bits = 0
compressed_bits = 0
restored_bit_planes = []

output_dir = "compressed_planes"
os.makedirs(output_dir, exist_ok=True)

# Сжатие и сохранение
for i in range(8):
    shape = bit_planes[i].shape
    bit_string = bit_plane_strings[i]

    compressed = lzw_compress(bit_string)
    packed_bytes = pack_12bit_codes(compressed)

    # Проверяем, что упаковка и распаковка работают корректно
    codes_after_unpack = unpack_12bit_codes(packed_bytes, len(compressed))

    print(f"Исходные первые 20 кодов:    {compressed[:20]}")
    print(f"Распакованные первые 20 кодов: {codes_after_unpack[:20]}")
    print(f"Длина исходных: {len(compressed)}, длина распакованных: {len(codes_after_unpack)}")

    for idx, (a, b) in enumerate(zip(codes_after_unpack, compressed)):
        if a != b:
            print(f"Первое отличие на индексе {idx}: распаковано {a}, ожидалось {b}")
            break
    else:
        print("Все элементы совпадают, возможно ошибка в длине списков.")

    assert codes_after_unpack == compressed, f"Ошибка на плоскости {i}: распакованные коды не совпадают с исходными!"

    assert codes_after_unpack == compressed, f"Ошибка на плоскости {i}: распакованные коды не совпадают с исходными!"
    print(f"Плоскость {i}: упаковка и распаковка кодов совпадают.")

    original_bits += len(bit_string)
    compressed_bits += len(compressed) * 12

    output_path = os.path.join(output_dir, f"plane_{i}.lzw")

    with open(output_path, 'wb') as f:
        f.write(struct.pack('II', *shape))
        f.write(struct.pack('I', len(compressed)))
        f.write(struct.pack('I', len(packed_bytes)))
        f.write(struct.pack('I', len(bit_string)))
        f.write(packed_bytes)

    decompressed = lzw_decompress(compressed)
    decompressed = decompressed[:len(bit_string)]
    restored = bit_string_to_image(decompressed, shape)
    restored_bit_planes.append(restored)

    is_equal = np.array_equal(bit_planes[i], restored)
    print(f"Битовая плоскость {i}: {'OK' if is_equal else 'Ошибка'}")

compression_ratio = original_bits / compressed_bits
print(f"\nКоэффициент сжатия: {compression_ratio:.2f}")

reconstructed_img = np.zeros_like(restored_bit_planes[0], dtype=np.uint8)
for i in range(8):
    reconstructed_img += (restored_bit_planes[i] << i)

cv2.imwrite(r"RestoredImages/reconstructed_image.png", reconstructed_img)
cv2.imwrite(r"RestoredImages/original_image.png", img)

total_compressed_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir))
original_size = img.size

print(f"Размер исходного изображения: {original_size} байт")
print(f"Общий размер .lzw файлов: {total_compressed_size} байт")

# Восстановление из файлов
restored_bit_planes_from_files = []

for i in range(8):
    path = os.path.join(output_dir, f"plane_{i}.lzw")
    with open(path, 'rb') as f:
        height, width = struct.unpack('II', f.read(8))
        length_compressed, = struct.unpack('I', f.read(4))
        packed_size, = struct.unpack('I', f.read(4))
        bit_length, = struct.unpack('I', f.read(4))
        data_bytes = f.read(packed_size)

    print(f"=== План {i} ===")
    print(f"Высота x Ширина: {height}x{width}")
    print(f"Компрессированные данные: {length_compressed}")
    print(f"Записанные упакованные байты: {packed_size}")
    print(f"Исходная длина строки: {bit_length}")

    codes = unpack_12bit_codes(data_bytes, length_compressed)
    decompressed_str = lzw_decompress(codes)
    decompressed_str = decompressed_str[:bit_length]

    try:
        restored_plane = bit_string_to_image(decompressed_str, (height, width))
        restored_bit_planes_from_files.append(restored_plane)
    except Exception as e:
        print(f"Ошибка на уровне {i}: {e}")

if len(restored_bit_planes_from_files) == 8:
    restored_img_from_files = np.zeros_like(restored_bit_planes_from_files[0], dtype=np.uint8)
    for i in range(8):
        restored_img_from_files += (restored_bit_planes_from_files[i] << i)

    cv2.imwrite("restored_from_files.png", restored_img_from_files)
    print("Восстановленное изображение из файлов сохранено как restored_from_files.png")

    if np.array_equal(img, restored_img_from_files):
        print("Восстановленное изображение точно совпадает с исходным!")
    else:
        print("Восстановленное изображение отличается от исходного.")
else:
    print("Не удалось восстановить все битовые плоскости.")