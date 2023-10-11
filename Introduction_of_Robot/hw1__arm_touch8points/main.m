clf
clear all

L1 = 0.35;
L2 = 0.305;
L3 = 0.3;
R = 0.2;
l = 0.4;

%建立机器人模型
%       theta       d       a       alpha   offset
L(1)=Link([0        0       0       0       0     ],'modified'); %改进D-H参数
L(2)=Link([0        L1      0       pi/2    0     ],'modified');
L(3)=Link([0        0       0       -pi/2   0     ],'modified');
L(4)=Link([0        L2      0       pi/2    0     ],'modified');
L(5)=Link([0        0       0       -pi/2   0     ],'modified');
L(6)=Link([0        L3      0       pi/2    0     ],'modified');
L(7)=Link([0        0       0       -pi/2   0     ],'modified');
robot=SerialLink(L,'name','ysq'); %连接连杆，机器人取名ysq
robot.plot([0 0 0 0 0 0 0])%机械臂图
robot.display()

target_points = [
    R,              l, 0
    -R,             l, 0
    0,              l, R
    0,              l, -R
    R*cos(pi/4),    l, R*sin(pi/4)
    R*cos(pi/4),    l, -R*sin(pi/4)
    -R*cos(pi/4),   l, R*sin(pi/4)
    -R*cos(pi/4),   l, -R*sin(pi/4)
    ];
hold on
plot3(target_points(:,1),target_points(:,2),target_points(:,3),"ro","LineWidth",2)

init_state = [0 0 0 0 0 0 0];

for i=1:8
    T = transl(target_points(i, :))
    inverse_kinematics = robot.ikine(T,'q0', init_state)
    t = 0:0.05:2;
    tra = jtraj(init_state, inverse_kinematics, t);
    robot.plot(tra);
    init_state = inverse_kinematics;
end
