import cv2
import numpy as np
import os
import struct

MAX_DICT_SIZE = 4096  # Максимальный размер словаря для LZW (12 бит)

# --- Функции упаковки/распаковки битов в байты и обратно ---

def bits_to_bytes(bits: str) -> bytes:
    b = bytearray()
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            byte += '0' * (8 - len(byte))  # дополняем нулями в конце
        b.append(int(byte, 2))
    return bytes(b)

def bytes_to_bits(data: bytes, bit_length: int) -> str:
    bits = bin(int.from_bytes(data, 'big'))[2:].zfill(len(data)*8)
    return bits[:bit_length]

class AdaptiveHuffmanNode:
    def __init__(self, symbol=None, weight=0, order=0):
        self.symbol = symbol
        self.weight = weight
        self.parent = None
        self.left = None
        self.right = None
        self.order = order

    def is_leaf(self):
        return self.left is None and self.right is None


class AdaptiveHuffmanTree:
    def __init__(self):
        self.max_order = 4095 * 2 + 1  # Достаточно большое число
        self.NYT = AdaptiveHuffmanNode(symbol=None, order=self.max_order)
        self.root = self.NYT
        self.nodes_by_symbol = {None: self.NYT}
        self.nodes_by_order = {self.max_order: self.NYT}
        self.leaves = {}

    def get_code(self, symbol):
        if symbol in self.leaves:
            return self._get_code_from_node(self.leaves[symbol])
        else:
            # Код для NYT + 12 бит нового символа
            code = self._get_code_from_node(self.NYT)
            code += format(symbol, '012b')
            return code

    def _get_code_from_node(self, node):
        code = ''
        while node != self.root:
            if node.parent.left == node:
                code = '0' + code
            else:
                code = '1' + code
            node = node.parent
        return code

    def update(self, symbol):
        if symbol not in self.leaves:
            # Добавить новый символ
            new_NYT = AdaptiveHuffmanNode(symbol=None, order=self.NYT.order - 2)
            new_leaf = AdaptiveHuffmanNode(symbol=symbol, weight=1, order=self.NYT.order - 1)
            internal = AdaptiveHuffmanNode(symbol=None, weight=1, order=self.NYT.order)

            internal.left = new_NYT
            internal.right = new_leaf
            new_NYT.parent = internal
            new_leaf.parent = internal

            if self.NYT.parent:
                if self.NYT.parent.left == self.NYT:
                    self.NYT.parent.left = internal
                else:
                    self.NYT.parent.right = internal
                internal.parent = self.NYT.parent
            else:
                self.root = internal

            self.NYT = new_NYT
            self.nodes_by_symbol[None] = new_NYT
            self.nodes_by_order[new_leaf.order] = new_leaf
            self.nodes_by_order[new_NYT.order] = new_NYT
            self.nodes_by_order[internal.order] = internal
            self.leaves[symbol] = new_leaf

            self._increment(internal.parent)
        else:
            self._increment(self.leaves[symbol])

    def _increment(self, node):
        while node:
            # Найти самый правый узел с тем же весом
            for order in sorted(self.nodes_by_order, reverse=True):
                candidate = self.nodes_by_order[order]
                if candidate.weight == node.weight and candidate != node and candidate != node.parent:
                    self._swap_nodes(node, candidate)
                    break
            node.weight += 1
            node = node.parent

    def _swap_nodes(self, a, b):
        # Обмен в дереве
        if a.parent == b or b.parent == a:
            return  # Не трогаем родителей

        a_parent, b_parent = a.parent, b.parent
        if a_parent.left == a:
            a_parent.left = b
        else:
            a_parent.right = b

        if b_parent.left == b:
            b_parent.left = a
        else:
            b_parent.right = a

        a.parent, b.parent = b_parent, a_parent

        # Обмен порядков
        a.order, b.order = b.order, a.order
        self.nodes_by_order[a.order] = a
        self.nodes_by_order[b.order] = b


class AdaptiveHuffmanCoder:
    def __init__(self):
        self.tree = AdaptiveHuffmanTree()

    def encode(self, data):
        bits = ''
        for symbol in data:
            bits += self.tree.get_code(symbol)
            self.tree.update(symbol)
        return bits

    def decode(self, bits):
        tree = AdaptiveHuffmanTree()
        result = []
        node = tree.root
        i = 0
        while i <= len(bits):
            if node.is_leaf():
                if node.symbol is None:
                    if i + 12 > len(bits):
                        break
                    sym = int(bits[i:i + 12], 2)
                    i += 12
                else:
                    sym = node.symbol

                result.append(sym)
                tree.update(sym)
                node = tree.root
            else:
                if i >= len(bits):
                    break
                if bits[i] == '0':
                    node = node.left
                else:
                    node = node.right
                i += 1
        return result

# --- LZW (по твоему коду, работа с битовыми строками '0'/'1') ---

def lzw_compress(data: str):
    dictionary = {'0':0, '1':1}
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
    dictionary = {0:'0', 1:'1'}
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
            raise ValueError(f"Invalid index {k}")
        result.append(entry)
        if dict_size < MAX_DICT_SIZE:
            dictionary[dict_size] = w + entry[0]
            dict_size += 1
        w = entry
    return ''.join(result)

def bit_string_to_image(bit_string, shape):
    flat = np.array([int(b) for b in bit_string[:shape[0]*shape[1]]], dtype=np.uint8)
    return flat.reshape(shape)

# --- ОСНОВНОЙ КОД ---

input_path = 'Images/green_1000x1000.tif'  # путь к цветному изображению
output_dir = 'compressed_color_planes/with_Huffman'
restored_dir = 'RestoredImages/with_Huffman'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(restored_dir, exist_ok=True)

img = cv2.imread(input_path, cv2.IMREAD_COLOR)
channels = cv2.split(img)
restored_channels = []

total_original_bits = 0
total_compressed_bits = 0

for ch_idx, channel in enumerate(channels):
    bit_planes = []
    bit_plane_strings = []
    for i in range(8):
        plane = ((channel >> i) & 1).astype(np.uint8)
        bit_planes.append(plane)
        flat = plane.flatten()
        bit_string = ''.join(str(b) for b in flat)
        bit_plane_strings.append(bit_string)

    restored_bit_planes = []
    channel_original_bits = 0
    channel_compressed_bits = 0
    print(f"Канал {ch_idx}:")

    # for i in range(8):
    #     shape = bit_planes[i].shape
    #     bit_string = bit_plane_strings[i]
    #     original_bits = len(bit_string)
    #
    #     # Сжимаем LZW
    #     compressed = lzw_compress(bit_string)
    #
    #     # Кодируем адаптивным Хаффманом
    #     coder = AdaptiveHuffmanCoder()
    #     encoded_bits = coder.encode(compressed)
    #     compressed_bits = len(encoded_bits)
    #
    #     # Упаковываем в байты (для записи в файл)
    #     encoded_bytes = bits_to_bytes(encoded_bits)
    #     compressed_bytes_len = len(encoded_bytes)
    #
    #     print(f"  Битовая плоскость {i}:")
    #     print(f"    Исходный размер: {original_bits} бит")
    #     print(f"    Сжатый размер (Адапт.Хаффман): {compressed_bits} бит, {compressed_bytes_len} байт")
    #     print(f"    Коэффициент сжатия: {original_bits / (compressed_bytes_len * 8):.3f}")
    #
    #     channel_original_bits += original_bits
    #     channel_compressed_bits += compressed_bits
    #
    #     # Запись и чтение файла, восстановление...
    #     out_path = os.path.join(output_dir, f"ch{ch_idx}_plane{i}.bin")
    #     with open(out_path, 'wb') as f:
    #         f.write(struct.pack('I', len(encoded_bits)))
    #         f.write(encoded_bytes)
    #
    #     with open(out_path, 'rb') as f:
    #         bit_len, = struct.unpack('I', f.read(4))
    #         encoded_data = f.read()
    #     encoded_bits_read = bytes_to_bits(encoded_data, bit_len)
    #
    #     # Новый объект адаптивного декодера
    #     decoder = AdaptiveHuffmanCoder()
    #     decoded = decoder.decode(encoded_bits_read)
    #
    #     # 👇 ВСТАВКА
    #     print(f"    🔎 Кол-во LZW-кодов до Хаффмана: {len(compressed)}")
    #     print(f"    🔎 Кол-во LZW-кодов после Хаффмана: {len(decoded)}")
    #     if len(decoded) != len(compressed):
    #         print("    ❗ Предупреждение: количество кодов не совпадает!")
    #
    #     if len(decoded) != shape[0] * shape[1]:
    #         print(f"⚠️ Предупреждение: длина декомпрессии {len(decoded)} != {shape[0] * shape[1]}")
    #
    #     decompressed = lzw_decompress(decoded)
    #     restored_plane = bit_string_to_image(decompressed, shape)
    #     restored_bit_planes.append(restored_plane)

    for i in range(8):
        shape = bit_planes[i].shape
        bit_string = bit_plane_strings[i]
        original_bits = len(bit_string)

        flat = bit_planes[i].flatten()
        if np.all(flat == 0):
            print(f"  Битовая плоскость {i} полностью нулевая — пропущена.")
            compressed_bits = 0
            channel_compressed_bits += compressed_bits

            # Создаем нулевую плоскость и добавляем
            restored_plane = np.zeros(shape, dtype=np.uint8)
            restored_bit_planes.append(restored_plane)
            continue

        # Сжимаем LZW
        compressed = lzw_compress(bit_string)

        # Кодируем адаптивным Хаффманом
        coder = AdaptiveHuffmanCoder()
        encoded_bits = coder.encode(compressed)
        compressed_bits = len(encoded_bits)

        encoded_bytes = bits_to_bytes(encoded_bits)
        compressed_bytes_len = len(encoded_bytes)

        print(f"  Битовая плоскость {i}:")
        print(f"    Исходный размер: {original_bits} бит")
        print(f"    Сжатый размер (Адапт.Хаффман): {compressed_bits} бит, {compressed_bytes_len} байт")
        print(f"    Коэффициент сжатия: {original_bits / (compressed_bytes_len * 8):.3f}")

        channel_original_bits += original_bits
        channel_compressed_bits += compressed_bits

        out_path = os.path.join(output_dir, f"ch{ch_idx}_plane{i}.bin")
        with open(out_path, 'wb') as f:
            f.write(struct.pack('I', len(encoded_bits)))
            f.write(encoded_bytes)

        with open(out_path, 'rb') as f:
            bit_len, = struct.unpack('I', f.read(4))
            encoded_data = f.read()
        encoded_bits_read = bytes_to_bits(encoded_data, bit_len)

        decoder = AdaptiveHuffmanCoder()
        decoded = decoder.decode(encoded_bits_read)

        if len(decoded) != len(compressed):
            print("    ❗ Предупреждение: количество кодов не совпадает!")

        decompressed = lzw_decompress(decoded)
        restored_plane = bit_string_to_image(decompressed, shape)
        restored_bit_planes.append(restored_plane)

    channel_ratio = channel_original_bits / (channel_compressed_bits if channel_compressed_bits else 1)
    print(f"Канал {ch_idx} общий коэффициент сжатия: {channel_ratio:.3f}\n")

    total_original_bits += channel_original_bits
    total_compressed_bits += channel_compressed_bits

    # Восстанавливаем канал из битовых плоскостей
    restored_channel = sum((restored_bit_planes[i] << i).astype(np.uint8) for i in range(8))
    restored_channels.append(restored_channel)

restored_img = cv2.merge(restored_channels)
cv2.imwrite(os.path.join(restored_dir, 'restored_image.tif'), restored_img)

original_size = img.nbytes
compressed_size = sum(
    os.path.getsize(os.path.join(output_dir, f))
    for f in os.listdir(output_dir)
    if f.endswith('.bin')
)

if np.array_equal(restored_img, img):
    print("Исходное и восстановленное изображения полностью совпадают после декодирования!", end="\n\n")

print(f"Размер оригинала: {original_size} байт")
print(f"Размер сжатых плоскостей: {compressed_size} байт")

ratio = original_size / compressed_size
print(f"Коэффициент сжатия: {ratio:.2f}")