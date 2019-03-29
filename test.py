from collections import Counter


obj = Counter()

obj["a"] = 1
obj["b"] = 343
obj["c"] = 132
obj["d"] = 423
obj["e"] = 21
obj["f"] = 25

print(obj)
for key in obj:
    print(key)

print("*" * 40)
print(obj.most_common(3))
