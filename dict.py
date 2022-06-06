import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# file = 'BMD_data.xlsx'
# sheets = ['25','26','27','28','29','30']
# modality = ['QCT','CALCBCT','QCBCT','CYCCBCT','UCBCT']

# indices = list(range(1,126))
# columns = ['31c','31t','41c','41t','34c','34t','44c','44t','36c','36t','46c','46t','36i','46i','21c','21t','11c','11t','24c','24t','14c','14t','26c','26t','16c','16t','26s','16s']


# for i in range(len(sheets)):
#     for j in range(len(modality)):
#         globals()[f'df_{sheets[i]}_{modality[j]}'] = pd.read_excel(file, sheet_name=sheets[i]).transpose().tail(125).iloc[:,(0+j*30):(28+j*30)]  # 와... globals 활용해서 for문으로 변수 자동 생성해버림 ㅋㅋㅋㅋ transpose 및 필요한 행 추출까지 바로 시행! +indexing으로 modality별 구분까지!!!
#         globals()[f'df_{sheets[i]}_{modality[j]}'].index = indices  
#         globals()[f'df_{sheets[i]}_{modality[j]}'].columns = columns    # index와 column명 정리!

# # print(df_28_UCBCT)
# # print(df_25_QCT)


sites = {'LAC':['31c','41c'], 'LAT':['31t','41t'], 'LPC':['34c','44c'], 'LPT':['34t','44t'], 'LMC':['36c','46c'], 'LMT':['36t','46t'], 'LI':['36i','46i'], 'UAC':['21c','11c'], 'UAT':['21t','11t'], 'UPC':['24c','14c'], 'UPT':['24t','14t'], 'UMC':['26c','16c'], 'UMT':['26t','16t'], 'US':['26s','16s']}


print(sites)

for key in sites:
    print(key)

# df_LAC_QCT = pd.concat([df_25_QCT['31c'], df_26_QCT['31c'], ..., df_25_QCT['41c']])

# print(df_LAC_QCT)

'''
for i in range(len(sheets)):
    print(globals()[f'df_{sheets[i]}']) # 와... globals 활용해서 for문으로 변수 자동 선택까지 가능!!!!!!!!!!
'''





# print(df[0])
# print(df[2])
# A = pd.concat([df[0],df[2]])
# print(A)
# B = pd.concat([df[1],df[3]])
# print(B)

# C = pd.concat([A,B], axis=1)
# print(C)

# print(df.iloc[:,[0,2]])

# sns.scatterplot(x='0', y='1', data=df_25)

# df.to_excel('df.xlsx')

# dataframe.plot(kind='scatter', x='QCT', y='QCBCT')
# plt.show()