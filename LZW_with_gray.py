import cv2
import numpy as np
import os
import struct

MAX_DICT_SIZE = 4096

# === УПАКОВКА / РАСПАКОВКА 12-битных КОДОВ ===
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
            code2 = ((b2 & 0xF) << 8) | b3
            codes.extend([code1, code2])
            i += 3
        elif i + 1 < length:
            b1 = data_bytes[i]
            b2 = data_bytes[i + 1]
            code1 = (b1 << 4) | (b2 >> 4)
            codes.append(code1)
            i += 2
        elif i < length:
            b1 = data_bytes[i]
            code1 = (b1 << 4)
            codes.append(code1)
            i += 1
        else:
            break

    return codes[:expected_count]

# === LZW ===

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
            raise ValueError(f"Неверный индекс: {k}")
        result.append(entry)
        if dict_size < MAX_DICT_SIZE:
            dictionary[dict_size] = w + entry[0]
            dict_size += 1
        w = entry

    return ''.join(result)

def bit_string_to_image(bit_string, shape):
    flat = np.array([int(b) for b in bit_string[:shape[0]*shape[1]]], dtype=np.uint8)
    return flat.reshape(shape)

# # === ПАРАМЕТРЫ ===
#
# input_path = 'Images/cup.tif'
# output_dir = 'compressed_gray_planes/without_Huffman'
# restored_dir = 'RestoredGrayImages'
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(restored_dir, exist_ok=True)
#
# # === СЖАТИЕ ===
#
# img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE) # Одноканальное изображение
#
# for i in range(8):
#     plane = ((img >> i) & 1).astype(np.uint8)
#     bit_string = ''.join(str(b) for b in plane.flatten())
#
#     compressed = lzw_compress(bit_string)
#     packed = pack_12bit_codes(compressed)
#
#     out_path = os.path.join(output_dir, f"plane{i}.lzw")
#     with open(out_path, 'wb') as f:
#         f.write(struct.pack('II', *plane.shape))  # height, width
#         f.write(struct.pack('I', len(compressed)))  # кол-во 12-битных кодов
#         f.write(struct.pack('I', len(packed)))     # длина packed байт
#         f.write(struct.pack('I', len(bit_string))) # длина строки бит
#         f.write(packed)
#
# # === ВОССТАНОВЛЕНИЕ ===
#
# restored_bit_planes = []
#
# for i in range(8):
#     in_path = os.path.join(output_dir, f"plane{i}.lzw")
#     with open(in_path, 'rb') as f:
#         height, width = struct.unpack('II', f.read(8))
#         shape = (height, width)
#
#         num_codes = struct.unpack('I', f.read(4))[0]
#         packed_len = struct.unpack('I', f.read(4))[0]
#         bit_string_len = struct.unpack('I', f.read(4))[0]
#
#         packed = f.read(packed_len)
#
#     unpacked = unpack_12bit_codes(packed, num_codes)
#     decompressed = lzw_decompress(unpacked)[:bit_string_len]
#     restored = bit_string_to_image(decompressed, shape)
#     restored_bit_planes.append(restored)
#
# # === СОБИРАЕМ ГРАДАЦИИ СЕРОГО ===
#
# restored_img = np.zeros_like(restored_bit_planes[0], dtype=np.uint8)
# for i in range(8):
#     restored_img += (restored_bit_planes[i] << i)
#
# cv2.imwrite(os.path.join(restored_dir, "reconstructed_gray.tif"), restored_img)
#
# # === ПРОВЕРКА ===
#
# if np.array_equal(img, restored_img):
#     print("Восстановленное изображение совпадает с исходным.")
# else:
#     print("Восстановленное изображение отличается от исходного.")
#
# original_size = img.nbytes
# compressed_size = sum(
#     os.path.getsize(os.path.join(output_dir, f))
#     for f in os.listdir(output_dir)
#     if f.endswith('.lzw')
# )
#
# print(f"Размер оригинала: {original_size} байт")
# print(f"Размер сжатых плоскостей: {compressed_size} байт")
#
# ratio = original_size / compressed_size
# print(f"Коэффициент сжатия: {ratio:.2f}")