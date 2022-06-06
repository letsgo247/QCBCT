import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file = 'BMD_data.xlsx'
df = pd.read_excel(file, sheet_name='25')
df = df.transpose()
df = df.tail(125)

print(df)
a = df[0]
print(a)

index = list(range(1,126))

column = ['31c', '31t',	'41c',	'41t',	'34c',	'34t',	'44c',	'44t',	'36c',	'36t',	'46c',	'46t',	'36i',	'46i',	'21c',	'21t',	'11c',	'11t',	'24c',	'24t',	'14c',	'14t',	'26c',	'26t',	'16c',	'16t',	'26s',	'16s']


# index = []
# for i in range(1,126):
#     index.append(f'QCBCT_{i}')

# print(index)

# a.index = index

# print(a)