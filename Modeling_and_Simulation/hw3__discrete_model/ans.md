## 1
离散时间系统规范即四元组 $<STATES, OUTPUTS, \delta_h, \lambda>$

其中：
- $STATES:(S_{-5}, S_{-4}, ..., S_i)$
- $OUTPUTS:(r_0, r_1, ..., r_i)$
- $\delta_h:S_i=a*S_{i-3}+b*S_{i-5}$
- $\lambda:r_i=S_i/m$

## 2
![[net.jpg]]

## 3
模型描述语句：
- $S_{i-1}=DELAY(S_{-1}, S_i)$
- $S_{i-2}=DELAY(S_{-2}, S_{i-1})$
- $S_{i-3}=DELAY(S_{-3}, S_{i-2})$
- $S_{i-4}=DELAY(S_{-4}, S_{i-3})$
- $S_{i-5}=DELAY(S_{-5}, S_{i-4})$
- $tmp_1=PROD(a, S_{i-3})$
- $tmp_2=PROD(b, S_{i-5})$
- $S_i=SUM(tmp_1, tmp_2)$
- $r_i=DIV(S_i, m)$

排序层次：
- 变量
    - $Lev(S_{i-3})=Lev(S_{i-5})=0$
    - $Lev(tmp_1)=Lev(tmp_2)=1$
    - $Lev(S_i)=2$
    - $Lev(r_i)=3$
- 语句
    - $Lev[tmp_1=PROD(a, S_{i-3})]=Lev(S_{i-3})=0$
    - $Lev[tmp_2=PROD(b, S_{i-5})]=Lev(S_{i-5})=0$
    - $Lev[S_i=SUM(tmp_1, tmp_2)]=\max\{Lev(tmp_1),Lev(tmp_2)\}=1$
    - $Lev[r_i=DIV(S_i, m)]=Lev(S_i)=2$
