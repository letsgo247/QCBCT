import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm


file = 'BMD_data.xlsx'
sheets = ['25','26','27','28','29','30']
# modality = ['QCT','QCBCT','CYC_CBCT','U_CBCT','CAL_CBCT']
modality = ['QCT','CAL_CBCT','QCBCT','CYC_CBCT','U_CBCT']   # 편의상 일단 기존 데이터 순서대로 추출 후 나중에 열 순서 바꾸는게 나을듯

indices = list(range(1,126))
columns = ['31c','31t','41c','41t','34c','34t','44c','44t','36c','36t','46c','46t','36i','46i','21c','21t','11c','11t','24c','24t','14c','14t','26c','26t','16c','16t','26s','16s']



# BMD_data.xlsx 에서 df_sheet_modal 추출 (ex. df_28_UCBCT)
for i in range(len(sheets)):
    for j in range(len(modality)):
        globals()[f'df_{sheets[i]}_{modality[j]}'] = pd.read_excel(file, sheet_name=sheets[i]).transpose().tail(125).iloc[:,(0+j*30):(28+j*30)]  # 와... globals 활용해서 for문으로 변수 자동 생성해버림 ㅋㅋㅋㅋ transpose 및 필요한 행 추출까지 바로 시행! +indexing으로 modality별 구분까지!!!
        # globals()[f'df_{sheets[i]}_{modality[j]}'].index = indices  
        globals()[f'df_{sheets[i]}_{modality[j]}'].columns = columns    # index와 column명 정리!

# print(df_28_UCBCT)
# print(df_25_QCT['31c'])



# final.xlsx 생성
#'''
sites = {'LAC':['31c','41c'], 'LAT':['31t','41t'], 'LPC':['34c','44c'], 'LPT':['34t','44t'], 'LMC':['36c','46c'], 'LMT':['36t','46t'], 'LI':['36i','46i'], 'UAC':['21c','11c'], 'UAT':['21t','11t'], 'UPC':['24c','14c'], 'UPT':['24t','14t'], 'UMC':['26c','16c'], 'UMT':['26t','16t'], 'US':['26s','16s']}


list = []

for site in sites:  # LAC
    globals()[f'list_{site}'] = []  # 빈 리스트 생성

    for modal in modality:  # LAC_QCT
        globals()[f'list_{site}_{modal}'] = []  # 빈 리스트 생성
        
        for value in sites[site]:   # 31c   
            for sheet in sheets:    # 25        
                globals()[f'list_{site}_{modal}'].append(globals()[f'df_{sheet}_{modal}'][value])   # LAC_QCT 리스트에다 df_25_QCT['31c'] append

        globals()[f'df_{site}_{modal}'] = pd.concat(globals()[f'list_{site}_{modal}'])  # 리스트를 세로로 concat서 한 column완성!

        globals()[f'list_{site}'].append(globals()[f'df_{site}_{modal}'])

    globals()[f'df_{site}'] = pd.concat(globals()[f'list_{site}'], axis=1).astype('float64')    # 리스트를 가로로 concat해서 한 블럭 완성!
    # regplot 에서 type 에러나서 float64로 미리 변환!

    indices = []

    for value in sites[site]:
        for sheet in sheets:
            for i in range(1,126):
                indices.append(f'{sheet}_{value}_{i:0>3}')  # index 부여 w/ i 오른쪽 정렬!

    globals()[f'df_{site}'].index = indices
    globals()[f'df_{site}'].columns = modality

    list.append(globals()[f'df_{site}'])

df = pd.concat(list)    # 모든 df_site를 세로로 concat해서 하나의 sheet로 합침

for index in df.index:
    df.loc[index,'patient'] = int(index[0:2])
    df.loc[index,'site'] = index[3:6]   # index 정보로부터 새로운 열 추가

df = df[['QCT','QCBCT','CYC_CBCT','U_CBCT','CAL_CBCT','patient','site']]    # 보기 편한 열 순서로 바꾸기!


writer = pd.ExcelWriter('data/final_6.xlsx', engine='openpyxl')
df.to_excel(writer)
writer.save()
#'''