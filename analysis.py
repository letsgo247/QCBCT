import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm


df = pd.read_excel('data/final_6.xlsx')
# print(df)
# print(df.max())
# print(df.min())

modality = ['QCBCT','CYC_CBCT','U_CBCT','CAL_CBCT']
full_modality = ['QCT','QCBCT','CYC_CBCT','U_CBCT','CAL_CBCT']
params = ['min', 'Max', 'mean', 'std']

sites = {'LAC':['31c','41c'], 'LAT':['31t','41t'], 'LPC':['34c','44c'], 'LPT':['34t','44t'], 'LMC':['36c','46c'], 'LMT':['36t','46t'], 'LI':['36i','46i'], 'UAC':['21c','11c'], 'UAT':['21t','11t'], 'UPC':['24c','14c'], 'UPT':['24t','14t'], 'UMC':['26c','16c'], 'UMT':['26t','16t'], 'US':['26s','16s']}    

muc = pd.MultiIndex.from_product([full_modality,params])   # multiindex 활용한 다층 culumn 생성
indices = [key for key in sites]
df_param = pd.DataFrame(index = indices, columns = muc)
# print('<df_param>: ', df_param) # min, Max, mean, std 출력




def analysis(group):
    df_list = []
    for site in sites[group]:
        # print('야임마', site)
        df_list.append(df[df['site']==site])
    df_sub = pd.concat(df_list)
   


    print(f'================================{group}================================')
    # print(df_sub)
    # print(df_sub['QCT'])
    for modal in full_modality:
        # print(df_param[modal,'min'][group])
        # print(df_sub[modal].min())
        df_param[modal,'min'][group] = df_sub[modal].min()  # multiindex 조회
        df_param[modal,'Max'][group] = df_sub[modal].max()
        df_param[modal,'min-Max'][group] = f'{df_sub[modal].min()}-{df_sub[modal].max()}'
        df_param[modal,'mean'][group] = round(df_sub[modal].mean(), 2)
        df_param[modal,'std'][group] = round(df_sub[modal].std(), 2)
        # df_param[modal,'mean'][group] = f'{df_sub[modal].mean():.2f}'


        # print('<min>', df_sub['QCT'].min()) # 최소
        # print('<max>', df_sub.max()) # 최대
        # print('<mean>', df_sub.mean())  # 평균
        # print('<std>', df_sub.std())   # 표준편차



    for modal in modality:
        # 아래는 선형회귀
        # '''
        plt.figure(figsize=(7,7))
        plt.xlim(-700,1300)
        plt.ylim(-700,1300)

        X = df_sub['QCT']
        Y = df_sub[modal]

        sns.regplot(x='QCT', y=modal, data=df_sub, ci=None, line_kws={'color':'red', 'alpha':0.9})    # df column 중에서 x랑 y 선택해서 plot / ci:confidence interval
        # 그래프: https://mindscale.kr/course/python-visualization-basic/relation/

        x = range(-1000,2000)
        y = x
        plt.plot(x,y,ls='dotted')


        # 추세선값 & R2: https://jimmy-ai.tistory.com/190
        fit_line = np.polyfit(X, Y, 1)
        # print('fit_line: ', fit_line)
        sign = '+' if fit_line[1] >= 0 else '-' # text 표시용 sign 미리 정하기

        est_Y = np.array(X) * fit_line[0] + fit_line[1]
        r2 = r2_score(Y, est_Y)
        # print('r2: ', r2)

        
        plt.xlabel('QCT $(mg/cm^3)$')
        plt.ylabel(f'{modal} $(mg/cm^3)$')

        plt.text(100,1100, f'$R^2$ = {r2:.3f}', size = 12)   # 위치는 그냥 x,y 단위 그대로 / 소수점 셋째자리까지
        plt.text(100,1000, f'y = {fit_line[0]:.2f}x {sign} {abs(fit_line[1]):.2f}', size=12)


        # 회귀분석: https://lovelydiary.tistory.com/339
        # print(ols(f'{modal} ~ QCT', data=df_sub).fit().summary())    # 종속(Y) ~ 독립(X)


        plt.savefig(f'result/linear_{group}_{modal}.png', bbox_inches='tight')
        plt.close()
        # plt.show()  # ㄷㄷㄷ 이게 필요하다고 하네
        # '''
            

        # 아래는 bland-altman
        # '''
        f,ax = plt.subplots(1, figsize=(7,7))
        sm.graphics.mean_diff_plot(X,Y, ax=ax)
        plt.xlim(-700,1300)
        plt.ylim(-1000,1000)
        
        plt.xlabel(f'mean of QCT & {modal} $(mg/cm^3)$')
        plt.ylabel(f'difference of QCT & {modal} $(mg/cm^3)$')

        plt.savefig(f'result/Bland_Altman_{group}_{modal}.png', bbox_inches='tight')
        plt.close()
        # plt.show()
        # '''


# 별개 출력
# analysis('LAC')


# 전체 출력
for group in sites:
    analysis(group)
        


print('<df_param>: ', df_param) # min, Max, mean, std 출력


writer = pd.ExcelWriter('data/df_param.xlsx', engine='openpyxl')
df_param.to_excel(writer)
writer.save()