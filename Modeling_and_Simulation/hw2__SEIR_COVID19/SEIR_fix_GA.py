#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
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
                 para = [0.6, 0.6, 0.25, 0.1, 0.12, 0.9, 0.05, 0.05**2], method='conpara'):
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
        para = [self.beta1[i], self.beta2[i], self.sigma[i], self.gamma1[i], self.gamma2[i], self.q[i], self.k1[i], self.k2[i]]
        print(para)
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

def cal_loss(para):
    y = Covid19_wh.import_overall_data(start='2020/4/1')#, end='2020/2/21')
    psum = Covid19_wh.total_population()
    x = [0, 0, y['confirmed_now'][0], 0, y['cured'][0], y['dead'][0]]
    x[0] = psum-x[1]-x[2]-x[3]-x[4]-x[5]
    people_t1 = [11170306.672189204, 41146.34483840334, 148.38424407665607, 332.672297733851, 47.3443995012965, 18.582031081383487] # 预测的封城时人数
    people_t2 = [10697861.609524352, 460820.02594214753, 4441.7605724521945, 42380.328518225324, 5043.572763593103, 1449.702679231047]
    people_t3 = [9505892.739656894, 1652788.260343106, 4139.182665828233, 33419.51182969577, 9357.309119035368, 6396.996385440625]
    x = [int(i) for i in people_t3]
    y = y.drop('confirmed', axis=1)
    t = np.linspace(1,len(y),len(y))
    para0 = [0.3, 0.6, 0.7, 0.1, 0.05, 0.75, 0.02, 0.02**2]
    model = SEIQRD(x, para)
    model.step(len(y)-1, 1)
    loss = (MSE(model.D, y['dead'])
            +MSE(np.array(model.I)+np.array(model.Q), y['confirmed_now'])
            +MSE(model.R, y['cured']))/3
    return loss


if __name__ == "__main__":
    para_t1 = [0.32844575, 0.31867058, 0.00391007, 0.06940371, 0.00684262, 0.68914956, 0.01075269, 0.00782014]  # 封城前参数1.23
    people_t1 = [11170306.672189204, 41146.34483840334, 148.38424407665607, 332.672297733851, 47.3443995012965, 18.582031081383487] # 预测的封城时人数
    para_t2 = [0.00000000e+00, 9.87292278e-02, 1.07526882e-02, 1.85728250e-02,
            1.07526882e-02, 9.90224829e-01, 2.24828935e-02, 9.77517107e-04]    #2.11
    people_t2 = [10697861.609524352, 460820.02594214753, 4441.7605724521945, 42380.328518225324, 5043.572763593103, 1449.702679231047]
    best_x=[1.  , 0.15053763, 0.  , 0.,   0.01270772, 0., 0.00782014, 0.01368524]   # 2.21
    people_t3 = [9505892.739656894, 1652788.260343106, 4139.182665828233, 33419.51182969577, 9357.309119035368, 6396.996385440625]
    ysq_para_t4 = [0., 0., 0., 0.02, 0.05, 0.1, 0.02, 0.02**2]  # 2.21-4.1
    people_t4 = [9505892.0, 1652788.0, 17.763336048518795, 5308.325273750826, 40753.69968264805, 7231.211707552596]
    para_t5 = [1., 0.9941349, 0. , 1. , 1.  , 0.75073314, 0. , 0. ]

    #ga = GA(func=cal_loss, n_dim=8, size_pop=50, max_iter=500, prob_mut=0.001, lb=[0]*8, ub=[1]*8, precision=1e-3)
    #ga.best_x = ysq_para_t4
    #best_x, best_y = ga.run()
    #print(f'best_x={best_x}, best_y={best_y}')

    y = Covid19_wh.import_overall_data()
    psum = Covid19_wh.total_population()
    para = Covid19_wh.import_para()
    x = [0, 0, y['confirmed_now'][0], 0, y['cured'][0], y['dead'][0]]
    x[0] = psum-x[1]-x[2]-x[3]-x[4]-x[5]
    y = y.drop('confirmed', axis=1)
    t = np.linspace(1,len(y),len(y))
    model = SEIQRD(x, para, method='varypara')
    model.vary_run(len(y)-1, 1)
    print(f'S:{model.S[-1]}, E:{model.E[-1]}, I:{model.I[-1]}, Q:{model.Q[-1]}, R:{model.R[-1]}, D:{model.D[-1]}')
    #plt.plot(t, model.E, label='E')
    plt.plot(t, model.I, label='I')
    plt.plot(t, model.Q, label='Q')
    plt.plot(t, model.R, label='R')
    plt.plot(t, model.D, label='D')
    plt.plot(t, np.array(model.Q)+np.array(model.I), label='C')
    plt.plot(t, y['confirmed_now'], label='tC')
    plt.plot(t, y['cured'], label='tR')
    plt.plot(t, y['dead'], label='tD')
    plt.legend(loc='best')
    plt.savefig(f'tmp.png')
    plt.clf()
