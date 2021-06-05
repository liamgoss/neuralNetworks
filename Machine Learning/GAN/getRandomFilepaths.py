import pandas as pd
import random
n = 17769
s = int(n / 2)
filename = 'filePaths.csv'
skip = sorted(random.sample(range(n),n-s))
df = pd.read_csv(filename, skiprows=skip)
#print(type(df))