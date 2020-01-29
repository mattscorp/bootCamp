from ft_filter import ft_filter
from ft_map import ft_map
from ft_reduce import ft_reduce

def addition(n):
    return n + n

def is_even(n):
    return n % 2

def machin(n,m):
    return n * m

tab = [1, 2, 3, 4]
tuple = (1, 2, 3, 4)
list = {1, 2, 3, 4}

print("MAP")
print(tab)
result = ft_map(addition, tab)
print(result)
# print(list(result))

print(tuple)
result = ft_map(addition, tuple)
print(result)
# print(list(result))

print(list)
result = ft_map(addition, list)
print(result)
# print(list(result))

print("FILTER")
print(tab)
result = ft_filter(is_even, tab)
print(result)
# print(list(result))

print(tuple)
result = ft_filter(is_even, tuple)
print(result)
# print(list(result))

print(list)
result = ft_filter(is_even, list)
print(result)
# print(list(result))

print("REDUCE")
print(tab)
result = ft_reduce(machin, tab)
print(result)
# print(list(result))

print(tuple)
result = ft_reduce(machin, tuple)
print(result)
# print(list(result))

print(list)
result = ft_reduce(machin, list)
print(result)
# print(list(result))
