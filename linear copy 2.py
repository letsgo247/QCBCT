import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file = 'BMD_data.xlsx'
df = pd.read_excel(file, sheet_name='25')
print(df)

# df = df.head(125)

# print(df.columns)
# print(df.index)

# print(df['31c'])
# print(df['41c'])
# result = pd.concat([df['31c'],df['41c']])
# print(result)

# print(result.loc[0])

# result.to_excel('result.xlsx')

# sns.scatterplot(x='31c', y='41c', data=df)