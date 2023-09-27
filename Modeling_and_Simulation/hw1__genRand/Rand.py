#!/usr/bin/env python
# coding=utf-8
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.utils.validation import check_is_fitted, check_array

class Uniform:
    # 选的c的rand的参数
    def __init__(self, 
                 seed : float = 345, 
                 M : float = 32768, 
                 c : float = 12345/65535, 
                 a : float = 1103515245/65535):
        self.x = seed
        self.M = M
        self.c = c
        self.a = a
        self.gen = self.step()

    def step(self):
        while 1:
            self.x = (self.a*self.x+self.c)%self.M
            yield self.x/self.M

    def run(self, n : int = 1) -> float | list[float]:
        #if n <= 1:
        #    return self.gen.__next__()
        #else:
        res = []
        for i in range(n):
            res.append(self.gen.__next__())
        return res

    def test(self, n : float = 100):
        '''测试周期和均匀程度（都不错'''
        a = self.run(n)
        sns.lineplot(a)
        plt.savefig('test_uni.png')
        plt.clf()
        a.sort()
        sns.lineplot(a)
        plt.savefig('test_uni2.png')
        plt.clf()

class PxSample:
    def __init__(self, 
                 p : callable, 
                 xrange : list[float], 
                 pmax : float):
        '''
        args:
            - p: 我们期望采样的概率分布
            - xrange: p的自变量的取值范围
            - f: 包住p的（先只采用01均匀分布，比较简单
        '''
        self.U = Uniform()
        self.p = p
        self.xrange = xrange.copy()
        self.pmax = pmax

    def run(self, n) -> np.ndarray[float]:
        '''生成服从p分布的n个采样点'''
        x = (np.array(self.U.run(math.ceil(n*self.pmax)))
             * (self.xrange[1]-self.xrange[0])+self.xrange[0])
        y = np.array(self.U.run(math.ceil(n*self.pmax)))*self.pmax
        yhat = self.p(x)
        res = x[y<yhat]
        nop = np.sum(y>=yhat)
        # 若接收个数小于n，则补齐
        for i in range(n-len(res)):
            x = np.array(self.U.run(1))*(self.xrange[1]-self.xrange[0])+self.xrange[0]
            y = np.array(self.U.run(1))*self.pmax
            if y<self.p(x):
                res = np.append(res, x)
            else:
                nop += 1
        print(f"仿真舍选效率={(n-nop)/n}")
        return res

# 概率密度函数，输入x为自变量list
def onefunc(x : np.ndarray[float]) -> np.ndarray[float]:
    x = check_array(x, dtype=np.float64, ensure_2d=False)
    mask = (x<0)+(x>1)
    y = (12 / ((3 + 2*(3**0.5)) * math.pi)
        * (math.pi/4 + 2*(3**0.5)/3 * (1-x**2)**0.5))
    y[mask] = 0
    return y

if __name__ == "__main__":
    U = Uniform()
    U.test()
    
    pmax = onefunc([0])
    P = PxSample(onefunc, xrange=[0,1], pmax=pmax)
    y = onefunc(np.linspace(0,1,100))

    sns.lineplot(x=np.linspace(0,1,100), y=y)
    plt.savefig('tmp.png')
    plt.clf()
    x = P.run(100)
    sns.distplot(x,bins=20)
    plt.savefig('tmp2.png')
    print(f"实际舍选效率={1/pmax[0]}")
