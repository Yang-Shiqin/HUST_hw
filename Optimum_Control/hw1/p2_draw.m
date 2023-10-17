% 画图：
% t=0到1+sqrt(3/2)时，x1=-0.5t^2+t+1, x2=-t+1
% t=1+sqrt(3/2)到1+sqrt(6)时，x1=0.5t^2+t+k, x2=t-sqrt(3/2)
% 定义参数范围
t1 = linspace(0, pi -2*asin(2/sqrt(5)), 100);
t2 = linspace(pi -2*asin(2/sqrt(5)), 2*pi -2*asin(2/sqrt(5)), 100);
k = -1;

% 计算函数值
x1_t1 = sqrt(5) .* sin(t1+asin(2/sqrt(5))) - 1;
x2_t1 = sqrt(5) .* cos(t1+asin(2/sqrt(5)));

x1_t2 = sin(t2 + 2*asin(2/sqrt(5))) + 1;
x2_t2 = cos(t2 + 2*asin(2/sqrt(5)));

% 绘制函数图形
figure;
hold on;
plot(t1, x1_t1, 'b', 'LineWidth', 2);
plot(t1, x2_t1, 'r', 'LineWidth', 2);
plot(t2, x1_t2, 'b', 'LineWidth', 2);
plot(t2, x2_t2, 'r', 'LineWidth', 2);
hold off;

xlabel('t');
ylabel('x');
legend('x1', 'x2');
grid on;
title('Function Plot');

% 绘制 x1 和 x2 的图形
figure;
plot(x1_t1, x2_t1, 'b', 'LineWidth', 2);
hold on;
plot(x1_t2, x2_t2, 'r', 'LineWidth', 2);
hold off;

xlabel('x1');
ylabel('x2');
legend('t=0 to 1+sqrt(3/2)', 't=1+sqrt(3/2) to 1+sqrt(6)');
grid on;
title('Function Plot');