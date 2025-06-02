import cv2
import numpy as np
import os
from collections import defaultdict, deque

MAX_DICT_SIZE = 4096
INIT_DICT_SIZE = 256

class BitWriter:
    def __init__(self):
        self.buffer = bytearray()
        self.current_byte = 0
        self.bit_count = 0

    def write_bits(self, bits, length):
        for i in range(length):
            bit = (bits >> (length - i - 1)) & 1
            self.current_byte = (self.current_byte << 1) | bit
            self.bit_count += 1
            if self.bit_count == 8:
                self.buffer.append(self.current_byte)
                self.current_byte = 0
                self.bit_count = 0

    def get_bytes(self):
        if self.bit_count > 0:
            self.current_byte <<= (8 - self.bit_count)
            self.buffer.append(self.current_byte)
        return bytes(self.buffer)

class BitReader:
    def __init__(self, data):
        self.buffer = deque(data)
        self.current_byte = 0
        self.bit_count = 0

    def read_bits(self, length):
        result = 0
        for _ in range(length):
            if self.bit_count == 0:
                if not self.buffer:
                    raise EOFError("Unexpected end of data")
                self.current_byte = self.buffer.popleft()
                self.bit_count = 8
            result = (result << 1) | ((self.current_byte >> (self.bit_count - 1)) & 1)
            self.bit_count -= 1
        return result

class Node:
    def __init__(self, symbol=None):
        self.symbol = symbol
        self.weight = 0
        self.parent = None
        self.left = None
        self.right = None
        self.number = 0

class FGKTree:
    def __init__(self):
        self.NYT = Node()
        self.root = self.NYT
        self.nodes = {None: self.NYT}
        self.symbol_nodes = {}
        self.max_number = 0

    def get_code(self, symbol):
        if symbol in self.symbol_nodes:
            return self._get_path(self.symbol_nodes[symbol])
        return self._get_path(self.NYT) + format(symbol, '012b')

    def _get_path(self, node):
        path = ''
        while node.parent:
            path = ('0' if node.parent.left == node else '1') + path
            node = node.parent
        return path

    def update(self, symbol):
        if symbol in self.symbol_nodes:
            self._increment(self.symbol_nodes[symbol])
        else:
            new_internal = Node()
            new_symbol = Node(symbol)
            new_internal.left = self.NYT
            new_internal.right = new_symbol
            new_internal.parent = self.NYT.parent
            new_symbol.parent = new_internal
            self.NYT.parent = new_internal

            if new_internal.parent:
                if new_internal.parent.left == self.NYT:
                    new_internal.parent.left = new_internal
                else:
                    new_internal.parent.right = new_internal
            else:
                self.root = new_internal

            self.symbol_nodes[symbol] = new_symbol
            self.nodes[None] = self.NYT
            self.nodes[symbol] = new_symbol

            self._assign_numbers()
            self._increment(new_internal.parent if new_internal.parent else new_internal)

    def _assign_numbers(self):
        def assign(node):
            if node:
                node.number = self.max_number
                self.max_number += 1
                assign(node.left)
                assign(node.right)
        self.max_number = 0
        assign(self.root)

    def _increment(self, node):
        while node:
            candidate = self._find_highest_node(node.weight)
            if candidate and candidate != node and candidate.parent != node:
                self._swap_nodes(candidate, node)
            node.weight += 1
            node = node.parent

    def _find_highest_node(self, weight):
        result = None
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            if node.weight == weight:
                result = node
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return result

    def _swap_nodes(self, a, b):
        a.symbol, b.symbol = b.symbol, a.symbol
        self.symbol_nodes[a.symbol], self.symbol_nodes[b.symbol] = a, b

    def decode(self, bit_str):
        node = self.root
        result = []
        i = 0
        while i < len(bit_str):
            while node.left and node.right:
                if bit_str[i] == '0':
                    node = node.left
                else:
                    node = node.right
                i += 1
            if node == self.NYT:
                symbol = int(bit_str[i:i+12], 2)
                i += 12
            else:
                symbol = node.symbol
            result.append(symbol)
            self.update(symbol)
            node = self.root
        return result

def lzw_compress(bits):
    dictionary = {format(i, '08b'): i for i in range(256)}
    dict_size = INIT_DICT_SIZE
    current = ''
    output = []
    for bit in bits:
        current += bit
        if current not in dictionary:
            if dict_size < MAX_DICT_SIZE:
                dictionary[current] = dict_size
                dict_size += 1
            output.append(dictionary[current[:-1]])
            current = bit
    if current:
        output.append(dictionary[current])
    return output

def lzw_decompress(codes):
    dictionary = {i: format(i, '08b') for i in range(256)}
    dict_size = INIT_DICT_SIZE
    result = ''
    prev = codes[0]
    result += dictionary[prev]
    for code in codes[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            entry = dictionary[prev] + dictionary[prev][0]
        else:
            raise ValueError("Invalid LZW code")
        result += entry
        if dict_size < MAX_DICT_SIZE:
            dictionary[dict_size] = dictionary[prev] + entry[0]
            dict_size += 1
        prev = code
    return result

def process_image(img):
    channels = cv2.split(img)
    bit_planes = []
    for ch in channels:
        for i in range(8):
            bit_planes.append(((ch >> i) & 1).astype(np.uint8))
    return bit_planes

def restore_image(bit_planes, shape):
    channels = []
    for c in range(3):
        ch = np.zeros(shape[:2], dtype=np.uint8)
        for i in range(8):
            ch |= (bit_planes[c * 8 + i] << i)
        channels.append(ch)
    return cv2.merge(channels)

def compress_and_decompress(path):
    img = cv2.imread(path)
    bit_planes = process_image(img)
    restored_planes = []
    for idx, plane in enumerate(bit_planes):
        bit_str = ''.join(plane.flatten().astype(str))
        codes = lzw_compress(bit_str)
        tree = FGKTree()
        encoded_bits = ''.join(tree.get_code(c) for c in codes)
        bit_writer = BitWriter()
        for bit in encoded_bits:
            bit_writer.write_bits(int(bit), 1)
        encoded_bytes = bit_writer.get_bytes()
        bit_reader = BitReader(encoded_bytes)
        decoded_bit_str = ''
        try:
            while True:
                decoded_bit_str += str(bit_reader.read_bits(1))
        except EOFError:
            pass
        tree = FGKTree()
        decoded_codes = tree.decode(decoded_bit_str)
        restored_bit_str = lzw_decompress(decoded_codes)
        restored_plane = np.array(list(restored_bit_str), dtype=np.uint8).reshape(plane.shape)
        restored_planes.append(restored_plane)
    restored_img = restore_image(restored_planes, img.shape)
    assert np.array_equal(img, restored_img)
    print("Сжатие и восстановление прошли успешно")

if __name__ == '__main__':
    compress_and_decompress('color_image.tif')