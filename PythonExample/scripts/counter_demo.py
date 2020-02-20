from collections import Counter

d = Counter()

d['a'] += 1
d['b'] += 1
d['c'] += 1
d['a'] += 1
print(d)

d['a'] = 1
d['b'] = 10
print(d)