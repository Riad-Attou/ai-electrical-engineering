%% Pendulum Param
initth = 0; % deg
L = 0.1;
mp = 0.05;
g = 9.81;

%% Motor Param
kt = 0.05;
ke = 0.05;
Lm = 4e-3;
R = 3;
b = 0.005;
J = 7.04e-5;

%% PID Param
Kp = 2;
Ki = 0.2;
Kd = 0.3;

a = (b/(b*R + ke*kt)) + ((J*(b*R+ke*kt)/(J*R))-b)/((b*R+ke*kt)/(J*R)^2);

p = (kt/(b*R + ke*kt))
c = (b*R+ke*kt)/(J*R)

d = 2*pi/p
ki = 2.1991/3.28