from vector import Vector

v1 = Vector([0.0, 1.0, 2.0, 3.0])
v2 = Vector([10.0, 11.0, 12.0, 10])
# print(v1)
Vlen = Vector([3, 2])# -> the vector will have values = [0.0, 1.0, 2.0]
# print(Vlen)
Vrange = Vector((10,15))# -> the vector will have values = [10.0, 11.0, 12.0,13.0, 14.0]
# print(Vrange)
print("ADD")
print(v1 + v2)
print(v1 + 2)
print(1 + v2)
print("SUB")
print(v1 - v2)
print(v1 - 2)
print(1 - v2)
print("DIV")
print(v1 / v2)
print(10 / v2)
print(0 / v2)
print(v2 / 0)
print(v2 / 10)
print("MUL")
print(v2 * v1)
print(v2 * 10)
print(10 * v2)
# print("Vres", vres)
# print("V2:", v2)
# vres = 3 * v2
# print("Vres", vres)
# vres = 1 - 2
# print(vres)
# v2 = v1 * 5
# print(v2)
