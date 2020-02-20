import pandas as pd
from collections import Counter


df = pd.read_csv("demo_data.csv", header=None)
nr, nc = df.shape
print("%d rows, %d columns data loaded." % (nr, nc))

df.columns = ['a', 'b', 'c', 'd', 'e']
d = Counter(df['a'].tolist())
print(d)
