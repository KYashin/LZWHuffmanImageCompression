# class AdaptiveHuffmanNode:
#     def __init__(self, symbol=None, weight=0, order=0):
#         self.symbol = symbol
#         self.weight = weight
#         self.parent = None
#         self.left = None
#         self.right = None
#         self.order = order
#
#     def is_leaf(self):
#         return self.left is None and self.right is None
#
#
# class AdaptiveHuffmanTree:
#     def __init__(self):
#         self.max_order = 4095 * 2 + 1  # Достаточно большое число
#         self.NYT = AdaptiveHuffmanNode(symbol=None, order=self.max_order)
#         self.root = self.NYT
#         self.nodes_by_symbol = {None: self.NYT}
#         self.nodes_by_order = {self.max_order: self.NYT}
#         self.leaves = {}
#
#     def get_code(self, symbol):
#         if symbol in self.leaves:
#             return self._get_code_from_node(self.leaves[symbol])
#         else:
#             # Код для NYT + 12 бит нового символа
#             code = self._get_code_from_node(self.NYT)
#             code += format(symbol, '012b')
#             return code
#
#     def _get_code_from_node(self, node):
#         code = ''
#         while node != self.root:
#             if node.parent.left == node:
#                 code = '0' + code
#             else:
#                 code = '1' + code
#             node = node.parent
#         return code
#
#     def update(self, symbol):
#         if symbol not in self.leaves:
#             # Добавить новый символ
#             new_NYT = AdaptiveHuffmanNode(symbol=None, order=self.NYT.order - 2)
#             new_leaf = AdaptiveHuffmanNode(symbol=symbol, weight=1, order=self.NYT.order - 1)
#             internal = AdaptiveHuffmanNode(symbol=None, weight=1, order=self.NYT.order)
#
#             internal.left = new_NYT
#             internal.right = new_leaf
#             new_NYT.parent = internal
#             new_leaf.parent = internal
#
#             if self.NYT.parent:
#                 if self.NYT.parent.left == self.NYT:
#                     self.NYT.parent.left = internal
#                 else:
#                     self.NYT.parent.right = internal
#                 internal.parent = self.NYT.parent
#             else:
#                 self.root = internal
#
#             self.NYT = new_NYT
#             self.nodes_by_symbol[None] = new_NYT
#             self.nodes_by_order[new_leaf.order] = new_leaf
#             self.nodes_by_order[new_NYT.order] = new_NYT
#             self.nodes_by_order[internal.order] = internal
#             self.leaves[symbol] = new_leaf
#
#             self._increment(internal.parent)
#         else:
#             self._increment(self.leaves[symbol])
#
#     def _increment(self, node):
#         while node:
#             # Найти самый правый узел с тем же весом
#             for order in sorted(self.nodes_by_order, reverse=True):
#                 candidate = self.nodes_by_order[order]
#                 if candidate.weight == node.weight and candidate != node and candidate != node.parent:
#                     self._swap_nodes(node, candidate)
#                     break
#             node.weight += 1
#             node = node.parent
#
#     def _swap_nodes(self, a, b):
#         # Обмен в дереве
#         if a.parent == b or b.parent == a:
#             return  # Не трогаем родителей
#
#         a_parent, b_parent = a.parent, b.parent
#         if a_parent.left == a:
#             a_parent.left = b
#         else:
#             a_parent.right = b
#
#         if b_parent.left == b:
#             b_parent.left = a
#         else:
#             b_parent.right = a
#
#         a.parent, b.parent = b_parent, a_parent
#
#         # Обмен порядков
#         a.order, b.order = b.order, a.order
#         self.nodes_by_order[a.order] = a
#         self.nodes_by_order[b.order] = b
#
#
# class AdaptiveHuffmanCoder:
#     def __init__(self):
#         self.tree = AdaptiveHuffmanTree()
#
#     def encode(self, data):
#         bits = ''
#         for symbol in data:
#             bits += self.tree.get_code(symbol)
#             self.tree.update(symbol)
#         return bits
#
#     def decode(self, bits):
#         tree = AdaptiveHuffmanTree()
#         result = []
#         node = tree.root
#         i = 0
#         while i <= len(bits):
#             if node.is_leaf():
#                 if node.symbol is None:
#                     if i + 12 > len(bits):
#                         break
#                     sym = int(bits[i:i + 12], 2)
#                     i += 12
#                 else:
#                     sym = node.symbol
#
#                 result.append(sym)
#                 tree.update(sym)
#                 node = tree.root
#             else:
#                 if i >= len(bits):
#                     break
#                 if bits[i] == '0':
#                     node = node.left
#                 else:
#                     node = node.right
#                 i += 1
#         return result
#
#
# # Пример использования
# coder = AdaptiveHuffmanCoder()
# original_data = [10, 20, 10, 3000, 4095, 20, 10]
# encoded_bits = coder.encode(original_data)
# decoded_data = coder.decode(encoded_bits)
#
# print("Original:", original_data)
# print("Decoded: ", decoded_data)
# print("Success: ", original_data == decoded_data)

import os

input_path = r'Images/cup.tif'
output_dir = "compressed_color_planes"

# Размер исходного файла на диске (лучше для сжатия)
original_size_bytes = os.path.getsize(input_path)

compressed_files = os.listdir(output_dir)
total_compressed_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in compressed_files)

compression_ratio = original_size_bytes / total_compressed_size

print(f'Коэффициент сжатия: {compression_ratio:.2f}')