#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 进度：目前只搭了SEIR基础的理论模型
# [ ] todo:
    # - [ ] 参数根据文献/数据确定
    # - [ ] 修正SEIR模型
    # - [ ] 搜集真实数据并代入进去
    # - [ ] 考虑人口流入流出


class SEIR:
    '''SEIR基础模型'''
    def __init__(self, init_seir=[100,0,1,0], 
                 beta1=0.6, beta2=0.6, sigma=0.25, gamma=0.1):
        self.beta1 = beta1      # 暴露者E传染给易感者S的接触率
        self.beta2 = beta2      # 感染者I传染给易感者S的接触率
        self.sigma = sigma      # 暴露者E变为感染者I的的感染率
        self.gamma = gamma      # 感染者I痊愈或死亡的概率

        initS, initE, initI, initR = init_seir
        self.S = np.array([int(initS)])     # 易感者
        self.E = np.array([int(initE)])     # 暴露者
        self.I = np.array([int(initI)])     # 感染者
        self.R = np.array([int(initR)])     # 移除者
        self.t = 0
        self.tseries = np.array([0])

    @staticmethod
    def model(t, seir_list, beta1, beta2, sigma, gamma):
        S, E, I, R = seir_list
        N = S+E+I+R

        dS = -beta1*S*I/N-beta2*S*E/N

        dE = -dS-sigma*E

        dI = sigma*E-gamma*I

        dR = gamma*I

        return [dS, dE, dI, dR]

    def step(self, runtime, dt=0.1):
        t_eval = np.arange(start=self.t, stop=self.t+runtime, step=dt)
        t_span = [self.t, self.t+runtime]
        seir_list = [self.S[-1], self.E[-1], self.I[-1], self.R[-1]]
        sol = solve_ivp(lambda t, y: SEIR.model(t, y, self.beta1, self.beta2, self.sigma, self.gamma),
                            t_span, seir_list, t_eval=t_eval)
        self.tseries = np.append(self.tseries, sol['t'])
        self.S = np.append(self.S, sol['y'][0])
        self.E = np.append(self.E, sol['y'][1])
        self.I = np.append(self.I, sol['y'][2])
        self.R = np.append(self.R, sol['y'][3])
        self.t = self.tseries[-1]

    def get(self):
        return self.S, self.E, self.I, self.R
        
    def plot(self):
        plt.plot(self.tseries, self.S, label='S')
        plt.plot(self.tseries, self.E, label='E')
        plt.plot(self.tseries, self.I, label='I')
        plt.plot(self.tseries, self.R, label='R')
        plt.legend(loc='best')
        plt.savefig('tmp.png')
        plt.clf()


if __name__ == "__main__":
    model = SEIR()
    model.step(50)
    model.plot()
