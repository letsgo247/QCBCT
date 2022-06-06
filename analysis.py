import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm


df = pd.read_excel('data/final_6.xlsx')
# df = pd.read_excel('data/comp_6.xlsx')
# patient_number = 30
# df = df[df['patient']==patient_number]

modality = ['QCBCT','CYC_CBCT','U_CBCT','CAL_CBCT']
full_modality = ['QCT','QCBCT','CYC_CBCT','U_CBCT','CAL_CBCT']
color_code = {'QCBCT':'red','CYC_CBCT':'blue','U_CBCT':'green','CAL_CBCT':'gold'}
params = ['min', 'Max', 'mean', 'std', 'slope']

sites = {
    'All':['31c','31t','41c','41t','34c','34t','44c','44t','36c','36t','46c','46t','36i','46i','21c','21t','11c','11t','24c','24t','14c','14t','26c','26t','16c','16t','26s','16s'],
    'Cortical':['31c','41c','34c','44c','36c','46c','21c','11c','24c','14c','26c','16c'],
    'Trabecular':['31t','41t','34t','44t','36t','46t','21t','11t','24t','14t','26t','16t'],
    'Maxillary':['21c','21t','11c','11t','24c','24t','14c','14t','26c','26t','16c','16t','26s','16s'],
    'Mandibular':['31c','31t','41c','41t','34c','34t','44c','44t','36c','36t','46c','46t','36i','46i'],
    'LAC':['31c','41c'], 'LAT':['31t','41t'], 'LPC':['34c','44c'], 'LPT':['34t','44t'], 'LMC':['36c','46c'], 'LMT':['36t','46t'], 'LI':['36i','46i'], 'UAC':['21c','11c'], 'UAT':['21t','11t'], 'UPC':['24c','14c'], 'UPT':['24t','14t'], 'UMC':['26c','16c'], 'UMT':['26t','16t'], 'US':['26s','16s']
    }    

muc = pd.MultiIndex.from_product([full_modality,params])   # multiindex 활용한 다층 culumn 생성
indices = [key for key in sites]
df_param = pd.DataFrame(index = indices, columns = muc)
print('<df_param>: ', df_param) # min, Max, mean, std 출력




def analysis(group):
    
    # 해당 group에 해당하는 site들 모아서 df_sub 생성
    df_list = []
    for site in sites[group]:
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
        # df_param[modal,'min-Max'][group] = f'{df_sub[modal].min()} - {df_sub[modal].max()}'
        df_param[modal,'mean'][group] = round(df_sub[modal].mean(), 2)
        df_param[modal,'std'][group] = round(df_sub[modal].std(), 2)

    
    df_param['QCT','slope'][group] = '-'    # 얘는 의미 없으므로 미리 넣어주기

    for modal in modality:
        # 아래는 선형회귀
        # '''
        plt.figure(figsize=(7,7))
        plt.xlim(-700,1300)
        plt.ylim(-700,1300)

        X = df_sub['QCT']
        Y = df_sub[modal]

        sns.regplot(x='QCT', y=modal, data=df_sub, ci=None, color=color_code[modal], scatter_kws={'alpha':0.2}, truncate=False)    # df column 중에서 x랑 y 선택해서 plot / ci:confidence interval
        # , line_kws={'color':'red', 'alpha':0.5}
        # 그래프: https://mindscale.kr/course/python-visualization-basic/relation/

        x = range(-1000,2000)
        y = x
        plt.plot(x,y,ls='dotted',color='black')   # y=x 기준선 그리기


        # 추세선값 & R2: https://jimmy-ai.tistory.com/190
        fit_line = np.polyfit(X, Y, 1)
        # print('fit_line: ', fit_line)
        df_param[modal,'slope'][group] = f'{fit_line[0]:.2f}'
        sign = '+' if fit_line[1] >= 0 else '-' # text 표시용 sign 미리 정하기

        est_Y = np.array(X) * fit_line[0] + fit_line[1]
        r2 = r2_score(Y, est_Y)
        # print('r2: ', r2)

        
        plt.xlabel('QCT $(mg/cm^3)$')
        plt.ylabel(f'{modal} $(mg/cm^3)$')

        plt.text(100,1100, f'y = {fit_line[0]:.2f}x {sign} {abs(fit_line[1]):.2f}', size=12)
        plt.text(100,1000, f'$R^2$ = {r2:.3f}', size = 12)   # 위치는 그냥 x,y 단위 그대로 / 소수점 셋째자리까지


        # 회귀분석: https://lovelydiary.tistory.com/339
        # print(ols(f'{modal} ~ QCT', data=df_sub).fit().summary())    # 종속(Y) ~ 독립(X)


        plt.savefig(f'result/linear_{group}_{modal}.png', bbox_inches='tight')
        # plt.savefig(f'result/{patient_number}/linear_{group}_{modal}.png', bbox_inches='tight')
        plt.close()
        # plt.show()  # ㄷㄷㄷ 이게 필요하다고 하네
        # '''
            



        # 아래는 bland-altman
        # '''
        f,ax = plt.subplots(1, figsize=(7,7))
        sm.graphics.mean_diff_plot(X,Y, ax=ax, scatter_kwds={'color':color_code[modal], 'alpha':0.5, 's':30})  # if size, 's'
        plt.xlim(-700,1300)
        plt.ylim(-1000,1000)
        
        plt.xlabel(f'mean of QCT & {modal} $(mg/cm^3)$')
        plt.ylabel(f'difference of QCT & {modal} $(mg/cm^3)$')

        plt.savefig(f'result/bland_Altman_{group}_{modal}.png', bbox_inches='tight')
        # plt.savefig(f'result/{patient_number}/Bland_Altman_{group}_{modal}.png', bbox_inches='tight')
        plt.close()
        # plt.show()
        # '''


# # 별개 출력
# analysis('LAC')


# 전체 출력
for group in sites:
    analysis(group)
        


print('<df_param>: ', df_param) # min, Max, mean, std 출력


# parameter 엑셀로 저장
writer = pd.ExcelWriter(f'data/df_param.xlsx', engine='openpyxl')
# writer = pd.ExcelWriter(f'data/{patient_number}/df_param_{patient_number}.xlsx', engine='openpyxl')
df_param.to_excel(writer)
writer.save()