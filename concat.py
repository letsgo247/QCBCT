import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file = 'BMD_data.xlsx'
df = pd.read_excel(file, sheet_name='25')
df = df.transpose()
df = df.tail(125)


print(df)
print(df[0])
print(df[2])
A = pd.concat([df[0],df[2]])
print(A)
B = pd.concat([df[1],df[3]])
print(B)

C = pd.concat([A,B], axis=1)
print(C)
