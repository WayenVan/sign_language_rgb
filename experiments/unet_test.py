import struct as s 

b = bytearray([1, 2, 3, 4000])

c = list(bytearray(b))
print(c[2])