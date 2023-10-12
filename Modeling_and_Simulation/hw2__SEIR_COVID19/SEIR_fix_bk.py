#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from sklearn.utils.validation import check_is_fitted, check_array

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
                 para = [0.6, 0.6, 0.25, 0.1, 0.12, 0.9, 0.05, 0.05**2]):
        self.beta1 = para[0]      # 暴露者E传染给易感者S的接触率
        self.beta2 = para[1]      # 感染者I传染给易感者S的接触率
        self.sigma = para[2]      # 暴露者E变为感染者I的的感染率
        self.gamma1 = para[3]     # 感染者I痊愈概率
        self.gamma2 = para[4]     # 隔离者Q痊愈概率
        self.q = para[5]          # 感染者I转换为隔离者Q的隔离率
        self.k1 = para[6]         # 感染者I死亡概率
        self.k2 = para[7]         # 隔离者Q死亡概率

#                 beta1=0.6, beta2=0.6, sigma=0.25, gamma1=0.1,
#                 gamma2=0.12, q=0.9, k1=0.05, k2=0.05*0.05):
#        self.beta1 = beta1      # 暴露者E传染给易感者S的接触率
#        self.beta2 = beta2      # 感染者I传染给易感者S的接触率
#        self.sigma = sigma      # 暴露者E变为感染者I的的感染率
#        self.gamma1 = gamma1    # 感染者I痊愈概率
#        self.gamma2 = gamma2    # 隔离者Q痊愈概率
#        self.q = q              # 感染者I转换为隔离者Q的隔离率
#        self.k1 = k1            # 感染者I死亡概率
#        self.k2 = k2            # 隔离者Q死亡概率

        initS, initE, initI, initQ, initR, initD = init_seir
        self.S = np.array([int(initS)])     # 易感者
        self.E = np.array([int(initE)])     # 暴露者
        self.I = np.array([int(initI)])     # 感染者
        self.Q = np.array([int(initQ)])     # 隔离者
        self.R = np.array([int(initR)])     # 康复者
        self.D = np.array([int(initD)])     # 死亡者
        self.t = 0
        self.tseries = np.array([0])

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

    def step(self, runtime, dt=0.1):
        t_eval = np.arange(start=self.t, stop=self.t+runtime, step=dt)
        t_span = [self.t, self.t+runtime]
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


def cal_loss(para, x, y):
    model = SEIQRD(x, para)
    model.step((len(y)//3)-1, 1)
    print((len(y)//3)-1)
    #loss = (MSE(model.D, y['dead'])
    #        +MSE(np.array(model.I)+np.array(model.Q), y['confirmed'])
    #        +MSE(model.R, y['cured']))
    a = np.array(y)
    b = np.array([*((np.array(model.Q)+np.array(model.I)).tolist()), *model.R, *model.D])
    print(a)
    print(f'b{b}')
    loss = a-b
    return loss

def Model(t, *para):
    model = SEIQRD([11211963, 0, 37, 0, 0, 0], np.exp(para))
    model.step(len(t)-1, 1)
    print(model.Q)
    return np.array([*model.R, *model.D, *((np.array(model.Q)+np.array(model.I)).tolist())])
    #return np.array([*((np.array(model.Q)+np.array(model.I)).tolist()), *model.R, *model.D])


if __name__ == "__main__":
    t = 4
    y = Covid19_wh.import_overall_data().iloc[:t]
    psum = Covid19_wh.total_population()
    x = [0, 0, y['confirmed_now'][0], 0, y['cured'][0], y['dead'][0]]
    x[0] = psum-x[1]-x[2]-x[3]-x[4]-x[5]
    t = np.linspace(1,t,t)
    y = y.drop('confirmed', axis=1)
    y = y.values.flatten().tolist()
    para0 = np.log([0.6, 0.6, 0.25, 0.1, 0.12, 0.9, 0.05, 0.05**2])
    #pFit, info = leastsq(cal_loss, np.array(para0), args=(x,y))
    pFit, _ = curve_fit(Model, t, y, p0=para0)
    print(pFit)
