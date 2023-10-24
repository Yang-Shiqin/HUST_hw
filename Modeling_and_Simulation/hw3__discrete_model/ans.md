## 1
离散时间系统规范即四元组 $<STATES, OUTPUTS, \delta_h, \lambda>$

其中：
- 状态变量集：$STATES:[0, m]\times[0, m]\times[0, m]\times[0, m]\times[0, m]$
    - 初始状态：$(S_{-5}, S_{-4}, S_{-3}, S_{-2}, S_{-1})$
- 转移函数：$\delta_h(S_{i-5}, S_{i-4}, S_{i-3}, S_{i-2}, S_{i-1}) = (S_{i-4}, S_{i-3}, S_{i-2}, S_{i-1}, (a*S_{i-3}+b*S_{i-5})%m)$
- $OUTPUTS:[0, 1]$
- $\lambda(S_{i-5}, S_{i-4}, S_{i-3}, S_{i-2}, S_{i-1})=[(a*S_{i-3}+b*S_{i-5})%m]/m$

## 2
![[net.jpg]]
少写了一个mod

## 3
模型描述语句：
- $S_{i-1}=DELAY(S_{-1}, S_i)$
- $S_{i-2}=DELAY(S_{-2}, S_{i-1})$
- $S_{i-3}=DELAY(S_{-3}, S_{i-2})$
- $S_{i-4}=DELAY(S_{-4}, S_{i-3})$
- $S_{i-5}=DELAY(S_{-5}, S_{i-4})$
- $tmp_1=PROD(a, S_{i-3})$
- $tmp_2=PROD(b, S_{i-5})$
- $tmp_3=SUM(tmp_1, tmp_2)$
- $S_i=MOD(tmp_3, m)$
- $r_i=DIV(S_i, m)$

排序层次：
- 变量
    - $Lev(S_{i-3})=Lev(S_{i-5})=0$
    - $Lev(tmp_1)=Lev(tmp_2)=1$
    - $Lev(tmp_3)=2$
    - $Lev(S_i)=3$
    - $Lev(r_i)=4$
- 语句
    - $Lev[tmp_1=PROD(a, S_{i-3})]=Lev(S_{i-3})=0$
    - $Lev[tmp_2=PROD(b, S_{i-5})]=Lev(S_{i-5})=0$
    - $Lev[tmp_3=SUM(tmp_1, tmp_2)]=\max\{Lev(tmp_1),Lev(tmp_2)\}=1$
    - $Lev[S_i=MOD(tmp_3, m)]=Lev(tmp_3)=2$
    - $Lev[r_i=DIV(S_i, m)]=Lev(S_i)=3$
