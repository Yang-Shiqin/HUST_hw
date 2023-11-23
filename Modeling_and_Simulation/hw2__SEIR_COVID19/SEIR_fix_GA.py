#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from sklearn.utils.validation import check_is_fitted, check_array
from sko.GA import GA

from helper import Covid19_wh

# 进度：目前只搭了SEIR基础的理论模型
# [ ] todo:
    # - [ ] 参数根据文献/数据确定
    # - [ ] 修正SEIR模型
    # - [ ] 搜集真实数据并代入进去
    # - [ ] 考虑人口流入流出


class SEIQRD:
    '''SEIR基础模型+修正'''
    def __init__(self, init_seir=[100,0,1,0,0,0], 
                 para = [0.6, 0.6, 0.25, 0.1, 0.12, 0.9, 0.05, 0.05**2], 
                 method='conpara', inout=None):
        self.inout = inout
        if method == 'conpara':
            self.conpara_init(para)
        else:
            self.varypara_init(para)

        initS, initE, initI, initQ, initR, initD = init_seir
        self.S = np.array([int(initS)])     # 易感者
        self.E = np.array([int(initE)])     # 暴露者
        self.I = np.array([int(initI)])     # 感染者
        self.Q = np.array([int(initQ)])     # 隔离者
        self.R = np.array([int(initR)])     # 康复者
        self.D = np.array([int(initD)])     # 死亡者
        self.t = 0
        self.tseries = np.array([0])

    def conpara_init(self, para):
        self.beta1 = para[0]      # 暴露者E传染给易感者S的接触率
        self.beta2 = para[1]      # 感染者I传染给易感者S的接触率
        self.sigma = para[2]      # 暴露者E变为感染者I的的感染率
        self.gamma1 = para[3]     # 感染者I痊愈概率
        self.gamma2 = para[4]     # 隔离者Q痊愈概率
        self.q = para[5]          # 感染者I转换为隔离者Q的隔离率
        self.k1 = para[6]         # 感染者I死亡概率
        self.k2 = para[7]         # 隔离者Q死亡概率

    def varypara_init(self, para):
        self.beta1 = para['beta1'].values       # 暴露者E传染给易感者S的接触率
        self.beta2 = para['beta2'].values       # 感染者I传染给易感者S的接触率
        self.sigma = para['sigma'].values       # 暴露者E变为感染者I的的感染率
        self.gamma1 = para['gamma1'].values     # 感染者I痊愈概率
        self.gamma2 = para['gamma2'].values     # 隔离者Q痊愈概率
        self.q = para['q'].values               # 感染者I转换为隔离者Q的隔离率
        self.k1 = para['k1'].values             # 感染者I死亡概率
        self.k2 = para['k2'].values             # 隔离者Q死亡概率

    @staticmethod
    def model(t, seir_list, beta1, beta2, sigma, gamma1, gamma2,
                 q, k1, k2):
        S, E, I, Q, R, D = seir_list
        N = S+E+I+Q+R+D

        dS = -beta1*S*I/N-beta2*S*E/N

        dE = -dS-sigma*E

        dI = sigma*E-gamma1*I-k1*I-q*I
        
        dQ = q*I-gamma2*Q-k2*Q

        dR = gamma1*I+gamma2*Q

        dD = k1*I+k2*Q

        return [dS, dE, dI, dQ, dR, dD]

    def conpara_step(self, runtime, dt=0.1):
        t_eval = np.arange(start=self.t, stop=self.t+runtime, step=dt)
        t_span = [self.t, self.t+runtime]
        i = int(self.t/dt)
        if self.inout is not None:
            self.S[-1] += self.inout['Inflow'][i]
            SER = self.S[-1]+self.E[-1]+self.R[-1]
            Srate = self.S[-1]/SER
            Erate = self.E[-1]/SER
            Sde = int(self.inout['Outflow'][i]*Srate)
            Ede = int(self.inout['Outflow'][i]*Erate)
            self.S[-1] -= Sde
            self.E[-1] -= Ede
            self.R[-1] -= self.inout['Outflow'][i]-(Sde+Ede)

        seir_list = [self.S[-1], self.E[-1], self.I[-1], self.Q[-1], self.R[-1], self.D[-1]]
        sol = solve_ivp(lambda t, y: SEIQRD.model(t, y, self.beta1, self.beta2, self.sigma, 
                                                self.gamma1, self.gamma2, self.q, self.k1, self.k2),
                            t_span, seir_list, t_eval=t_eval)
        self.tseries = np.append(self.tseries, sol['t'])
        self.S = np.append(self.S, sol['y'][0])
        self.E = np.append(self.E, sol['y'][1])
        self.I = np.append(self.I, sol['y'][2])
        self.Q = np.append(self.Q, sol['y'][3])
        self.R = np.append(self.R, sol['y'][4])
        self.D = np.append(self.D, sol['y'][5])
        self.t = self.tseries[-1]

    def varypara_step(self, dt=0.1):
        t_eval = np.arange(start=self.t, stop=self.t+dt, step=dt)
        t_eval = [self.t, self.t+dt]
        t_span = [self.t, self.t+dt]
        i = int(self.t/dt)
        if self.inout is not None:
            self.S[-1] += self.inout['Inflow'][i]
            SER = self.S[-1]+self.E[-1]+self.R[-1]
            Srate = self.S[-1]/SER
            Erate = self.E[-1]/SER
            Sde = int(self.inout['Outflow'][i]*Srate)
            Ede = int(self.inout['Outflow'][i]*Erate)
            self.S[-1] -= Sde
            self.E[-1] -= Ede
            self.R[-1] -= self.inout['Outflow'][i]-(Sde+Ede)
        para = [self.beta1[i], self.beta2[i], self.sigma[i], self.gamma1[i], self.gamma2[i], self.q[i], self.k1[i], self.k2[i]]
        seir_list = [self.S[-1], self.E[-1], self.I[-1], self.Q[-1], self.R[-1], self.D[-1]]
        sol = solve_ivp(lambda t, y: SEIQRD.model(t, y, *para), t_span, seir_list, t_eval=t_eval)
        self.tseries = np.append(self.tseries, sol['t'][-1])
        self.S = np.append(self.S, sol['y'][0][-1])
        self.E = np.append(self.E, sol['y'][1][-1])
        self.I = np.append(self.I, sol['y'][2][-1])
        self.Q = np.append(self.Q, sol['y'][3][-1])
        self.R = np.append(self.R, sol['y'][4][-1])
        self.D = np.append(self.D, sol['y'][5][-1])
        self.t = self.tseries[-1]

    def vary_run(self, runtime, dt=0.1):
        for i in range(int(runtime/dt)):
            self.varypara_step(dt)


    def get(self):
        return self.S, self.E, self.I, self.Q, self.R, self.D
        
    def plot(self):
        plt.plot(self.tseries, self.S, label='S')
        plt.plot(self.tseries, self.E, label='E')
        plt.plot(self.tseries, self.I, label='I')
        plt.plot(self.tseries, self.Q, label='Q')
        plt.plot(self.tseries, self.R, label='R')
        plt.plot(self.tseries, self.D, label='D')
        plt.legend(loc='best')
        plt.savefig('tmp.png')
        plt.clf()

def MSE(y, yhat):
    y = check_array(y, accept_sparse=False, dtype=np.float64, force_all_finite='allow-nan', ensure_2d=False)
    yhat = check_array(yhat, accept_sparse=False, dtype=np.float64, force_all_finite='allow-nan', ensure_2d=False)
    return ((y-yhat)**2).sum()/y.size

def RMSE(y, yhat):
    y = check_array(y, accept_sparse=False, dtype=np.float64, force_all_finite='allow-nan', ensure_2d=False)
    yhat = check_array(yhat, accept_sparse=False, dtype=np.float64, force_all_finite='allow-nan', ensure_2d=False)
    return np.sqrt(((y-yhat)**2).sum()/y.size)

def MAE(y, yhat):
    y = check_array(y, accept_sparse=False, dtype=np.float64, force_all_finite='allow-nan', ensure_2d=False)
    yhat = check_array(yhat, accept_sparse=False, dtype=np.float64, force_all_finite='allow-nan', ensure_2d=False)
    return (np.abs(y-yhat)).sum()/y.size

def R2(y, yhat):
    y = check_array(y, accept_sparse=False, dtype=np.float64, force_all_finite='allow-nan', ensure_2d=False)
    yhat = check_array(yhat, accept_sparse=False, dtype=np.float64, force_all_finite='allow-nan', ensure_2d=False)
    return 1-(((y-yhat)**2).sum()/y.size)/np.var(y)

period = [
    ['2019/12/27', '2020/1/23'],
    ['2020/1/23', '2020/2/11'],
    ['2020/2/11', '2020/2/21'],
    ['2020/2/21', '2020/3/31'],
    ['2020/3/31', '2020/6/27'],
]
choose = 4


# x，choose要改
def cal_loss(para):
    y = Covid19_wh.import_overall_data(*period[choose])
    psum = Covid19_wh.total_population()
    inout = Covid19_wh.import_inout_data(*period[choose])
    inout = pd.merge(y, inout, on='date', how='outer').fillna(0)
    x = [0, 0, y['confirmed_now'][0], 0, y['cured'][0], y['dead'][0]]
    x[0] = psum-x[1]-x[2]-x[3]-x[4]-x[5]
    x = [11245665.608168121, 6131.061161755346, 161.3192578983416, 294.25882264048136, 43.21752990286252, 21.535059682322032]
    x = [10934024.04055615, 35331.59565882226, 4156.139097055775, 12943.187056595178, 1127.7766617195962, 871.260969657528]
    x = [10934023.754434245, 3995.312486537398, 1401.6800345940985, 37771.74408315745, 7954.614450265398, 3304.8945112020083]
    x = [10934008.119882563, 2315.78050890061, 104.42501006266886, 4622.067458524575, 44093.60713994823, 3304.0]
    # x = [int(i) for i in ]
    y = y.drop('confirmed', axis=1)
    t = np.linspace(1,len(y),len(y))
    para0 = [0.3, 0.6, 0.7, 0.1, 0.05, 0.75, 0.02, 0.02**2]
    model = SEIQRD(x, para, method='conpara', inout=inout)
    model.conpara_step(len(y)-1, 1)
    loss = (MSE(model.D, y['dead'])
            +MSE(np.array(model.I)+np.array(model.Q), y['confirmed_now'])
            +MSE(model.R, y['cured']))/3
    return loss

tiaocan = 0
if __name__ == "__main__":
    psum = Covid19_wh.total_population()                            # 武汉初始总人口
    if tiaocan == 1:
        para_t0 = [1.29883178e-01, 2.84294086e-01, 2.10460437e-02, 2.98023242e-07, 2.68278138e-02, 5.33330472e-01, 8.76188330e-06, 1.33645542e-02]
        people1 = [11245665.608168121, 6131.061161755346, 161.3192578983416, 294.25882264048136, 43.21752990286252, 21.535059682322032]
        para_t1 = [0.50935099, 0.10827435, 0.0642097,  0.00200087, 0.01218247, 0.43582233, 0.00469226, 0.0083012]
        people2 = [10934024.04055615, 35331.59565882226, 4156.139097055775, 12943.187056595178, 1127.7766617195962, 871.260969657528]
        para_t2 = [0.00000000e+00, 1.90734875e-06, 2.42185667e-01, 1.78813945e-07, 2.52838150e-02, 9.31142028e-01, 1.19209297e-07, 9.01311690e-03]
        people3 = [10934023.754434245, 3995.312486537398, 1401.6800345940985, 37771.74408315745, 7954.614450265398, 3304.8945112020083]
        para_t3 = [0.00000000e+00, 1.27792366e-04, 1.44770750e-02, 4.76837187e-07, 6.01574814e-02, 3.35439166e-01, 0.00000000e+00, 0.00000000e+00]
        people4 = [10934008.119882563, 2315.78050890061, 104.42501006266886, 4622.067458524575, 44093.60713994823, 3304.0]
        para_t4 = [2.73455398e-02, 5.85975682e-02, 0.00000000e+00, 2.48134151e-04, 3.63220713e-01, 7.87442970e-03, 2.04086316e-04, 1.36296757e-01]

        ga = GA(func=cal_loss, n_dim=8, size_pop=50, max_iter=500, prob_mut=0.001, lb=[0]*8, ub=[1]*8, precision=1e-7)
        # ga.best_x = ysq_para_t4
        best_x, best_y = ga.run()
        print(f'best_x={best_x}, best_y={best_y}')

        y = Covid19_wh.import_overall_data(*period[choose])                            # 各种人数真实数据
        inout = Covid19_wh.import_inout_data(*period[choose])                          # 武汉封城前流入流出人口
        inout = pd.merge(y, inout, on='date', how='outer').fillna(0)    # 合并时间
        para = best_x 
        x = [0, 0, y['confirmed_now'][0], 0, y['cured'][0], y['dead'][0]]
        x[0] = psum-x[1]-x[2]-x[3]-x[4]-x[5]
        x = people4
        y = y.drop('confirmed', axis=1)
        model = SEIQRD(x, para, inout=inout)
        model.conpara_step(len(y)-1, 1)
    else:
        y = Covid19_wh.import_overall_data()                            # 各种人数真实数据
        inout = Covid19_wh.import_inout_data()                          # 武汉封城前流入流出人口
        inout = pd.merge(y, inout, on='date', how='outer').fillna(0)    # 合并时间
        x = [0, 0, y['confirmed_now'][0], 0, y['cured'][0], y['dead'][0]]
        x[0] = psum-x[1]-x[2]-x[3]-x[4]-x[5]
        y = y.drop('confirmed', axis=1)
        para = Covid19_wh.import_para()                                 # 拟合后参数
        model = SEIQRD(x, para, method='varypara', inout=inout)
        model.vary_run(len(y)-1, 1)
        df = pd.DataFrame(np.array([model.S, model.E, model.I, model.Q, model.R, model.D]).T, columns=['S', 'Q', 'I', 'Q', 'R', 'D'])
        df.to_csv('data/output.csv', index=False)

    t = np.linspace(1,len(y),len(y))
    print(f'S:{model.S[-1]}, E:{model.E[-1]}, I:{model.I[-1]}, Q:{model.Q[-1]}, R:{model.R[-1]}, D:{model.D[-1]}')
    #plt.plot(t, model.E, label='E')
    plt.plot(t, model.I, label='I')
    plt.plot(t, model.Q, label='Q')
    plt.plot(t, model.R, label='R')
    plt.plot(t, model.D, label='D')
    plt.plot(t, np.array(model.Q)+np.array(model.I), label='C')
    plt.plot(t, y['confirmed_now'], linestyle='--', label='tC')
    plt.plot(t, y['cured'], linestyle='-.', label='tR')
    plt.plot(t, y['dead'], linestyle=':', label='tD')
    plt.legend(loc='best')
    plt.savefig(f'tmp.png')
    plt.clf()

    mse = (MSE(model.D, y['dead'])
            +MSE(np.array(model.I)+np.array(model.Q), y['confirmed_now'])
            +MSE(model.R, y['cured']))/3
    mae = (MAE(model.D, y['dead'])
            +MAE(np.array(model.I)+np.array(model.Q), y['confirmed_now'])
            +MAE(model.R, y['cured']))/3
    rmse = (RMSE(model.D, y['dead'])
            +RMSE(np.array(model.I)+np.array(model.Q), y['confirmed_now'])
            +RMSE(model.R, y['cured']))/3
    r2 = (R2(model.D, y['dead'])
            +R2(np.array(model.I)+np.array(model.Q), y['confirmed_now'])
            +R2(model.R, y['cured']))/3
    print(f'MSE:{mse}, MRSE:{rmse}, MAE:{mae}, R2:{r2}')
