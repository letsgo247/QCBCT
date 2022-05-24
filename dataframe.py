import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm


file = 'BMD_data.xlsx'
sheets = ['25','26','27','28','29','30']
modality = ['QCT','CALCBCT','QCBCT','CYCCBCT','UCBCT']

indices = list(range(1,126))
columns = ['31c','31t','41c','41t','34c','34t','44c','44t','36c','36t','46c','46t','36i','46i','21c','21t','11c','11t','24c','24t','14c','14t','26c','26t','16c','16t','26s','16s']

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

writer = pd.ExcelWriter('final_3.xlsx', engine='openpyxl')

for site in sites:  # LAC
    globals()[f'list_{site}'] = []  # 빈 리스트 생성

    for modal in modality:  # QCT
        globals()[f'list_{site}_{modal}'] = []  # 빈 리스트 생성
        
        for value in sites[site]:   # 31c   
            for sheet in sheets:    # 25        
                globals()[f'list_{site}_{modal}'].append(globals()[f'df_{sheet}_{modal}'][value])   # 리스트에다 df_25_QCT['31c'] append

        globals()[f'df_{site}_{modal}'] = pd.concat(globals()[f'list_{site}_{modal}'])  # 리스트를 하나로 concat해서 완성!

        globals()[f'list_{site}'].append(globals()[f'df_{site}_{modal}'])

    globals()[f'df_{site}'] = pd.concat(globals()[f'list_{site}'], axis=1).astype('float64')    # regplot 에서 type 에러나서 float64로 미리 변환!

    indices = []

    for value in sites[site]:
        for sheet in sheets:
            for i in range(1,126):
                indices.append(f'{sheet}_{value}_{i:0>3}')  # i 오른쪽 정렬!

    globals()[f'df_{site}'].index = indices
    globals()[f'df_{site}'].columns = modality

    globals()[f'df_{site}'].to_excel(writer, sheet_name=site)

writer.save()
#'''



# 아래는 bland-altman
'''
X = df_LAC['QCT']
Y = df_LAC['QCBCT']

f,ax = plt.subplots(1, figsize=(7,7))
plt.xlim(-100,1500)
plt.ylim(-1000,1000)
sm.graphics.mean_diff_plot(X,Y, ax=ax)
plt.savefig('Bland_Altman.png')
plt.show()
'''



# 아래는 선형회귀
'''
plt.figure(figsize=(7,7))
plt.xlim(-100,1500)
plt.ylim(-100,1500)

# sns.scatterplot(x='QCT', y='QCBCT', data=df_LAC)    # df column 중에서 x랑 y 선택해서 plot
sns.regplot(x='QCT', y='QCBCT', data=df_LAC, ci=None, line_kws={'color':'red', 'alpha':0.9})    # df column 중에서 x랑 y 선택해서 plot / ci:confidence interval
# 그래프: https://mindscale.kr/course/python-visualization-basic/relation/

plt.xlabel('QCT $(mg/cm^3)$')
plt.ylabel('QCBCT $(mg/cm^3)$')


# x = range(-1000,2000)
# y = x
# plt.plot(x,y,ls='dotted',color='pink')


fit_line = np.polyfit(X, Y, 1)
print(fit_line)
sign = '+' if fit_line[1] >= 0 else '-' # text 표시용 sign 미리 정하기

est_Y = np.array(X) * fit_line[0] + fit_line[1]
r2 = r2_score(Y, est_Y)
print(r2)
# 추세선값&R2: https://jimmy-ai.tistory.com/190

plt.text(100,1100, f'$R^2$ = {r2:.3f}', size = 12)   # 위치는 그냥 x,y 단위 그대로 / 소수점 셋째자리까지
plt.text(100,1000, f'y = {fit_line[0]:.2f}x {sign} {abs(fit_line[1]):.2f}', size=12)

print(ols('QCBCT ~ QCT', data=df_LAC).fit().summary())    # 종속(Y) ~ 독립(X)
# 회귀분석: https://lovelydiary.tistory.com/339

plt.savefig('linear'.png')
plt.show()  # ㄷㄷㄷ 이게 필요하다고 하네
'''