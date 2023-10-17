%% 主函数
function main_Optimalcpntrol
%clc;close all;
T = 20; N = 1000; dt = T/N; M = 1;
solinit = bvpinit(linspace(0, T, N), [0 0 1 1]);
options = bvpset('Stats','on','RelTol',1e-1);
sol = bvp5c(@BVP_ode,@BVP_bc,solinit,options);
t = sol.x;
y = sol.y;

%% 最小化性能指标
X(1,:)=[1;1];
J=0;
for k=1:N-1
    if y(4,k)>0 
        U = -M;
    else
        U = M;
    end
    X(k+1,2) = X(k,2) + dt*U - dt*X(k,1);
    u(1,k)   = (X(k+1,2)-X(k,2)) / dt + X(k,1);
    X(k+1,1) = X(k,1) + dt*X(k,2);
    r        = dt;
    J        = J+r;
end

%% 画图
figure('Color',[1,1,1]);
plot(linspace(0,T,N), X(:,1), '-','LineWidth',1.5);hold on;
plot(linspace(0,T,N), X(:,2), '-.','LineWidth',1.5);
plot(linspace(0,T,999),u,'--','LineWidth',1.5);
axis([0 2 -2.5 2]);
xlabel('Time',...
       'FontWeight','bold');
ylabel('States',...
       'FontWeight','bold');
legend('Pos','Vel','Acc',...
       'LineWidth',1,...
       'EdgeColor',[1,1,1],...
       'Orientation','horizontal',...
       'Position',[0.5,0.93,0.40,0.055]);
set(gca,'FontName','Times New Roman',...
        'FontSize',15,...
        'LineWidth',1.2);
%     'YTick',[-2.5:1:2.1]);
saveas(gcf,'fg1.png');
end

%% 微分方程组
function dydt = BVP_ode(t,y)
M = 1;
if y(4)>0   
    u = -M;
else
    u = M;
end
dydt = [y(2)
        u-y(1)
        y(4)
        -y(3)];
end

%% 边界条件函数
function res = BVP_bc(ya,yb)
res = [ya(1) - 1
       ya(2) - 1
       yb(1)
       yb(2)];
end

