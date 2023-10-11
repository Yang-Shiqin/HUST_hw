#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 武汉convid数据集
class Covid19_wh:
    def __init__(self):
        pass

    # 获取pd.dataframe的武汉19/12/01~20/12/8的感染、治愈、死亡累计人数+当前感染
    @staticmethod
    def import_overall_data(start:str='2019/12/27', end:str='2020/6/27'):
        df = pd.read_csv('./data/overall_wuhan.csv')
        df = df[['date','confirmed','cured','dead']]
        df['date'] = pd.to_datetime(df['date'])
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        df = df[(df['date'] >= start) & (df['date'] <= end)]
        df['confirmed_now'] = df['confirmed']-df['cured']-df['dead']
        df.set_index('date', inplace=True)
        return df
    
    @staticmethod
    def import_inout_data(start:str='2019/12/27', end:str='2020/1/23'):
        df = pd.read_csv('./data/inout_wuhan.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        df = df[(df['Date'] >= start) & (df['Date'] <= end)]
        df.set_index('Date', inplace=True)
        return df
    
    @staticmethod
    def total_population():
        return 11212000 # 19年末
    
    @staticmethod
    def draw_data():
        df = Covid19_wh.import_overall_data()
        sns.lineplot(data=df)
        plt.axvline(x=pd.to_datetime('2020/1/23'), linestyle='--', color='blue')
        plt.savefig('tmp.png')
        plt.clf()
    
#print(Covid19_wh.import_overall_data())