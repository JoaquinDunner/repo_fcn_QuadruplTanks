clc

% Valores constante del laboratorio original
Ai = [28, 32, 28, 32]; % cm^2
ai = [0.071, 0.057, 0.071, 0.057]; % cm^2
g = 981; % cm/s^2
rho = 1; % g/cm^3
k = [3.33, 3.33];
kc = 1; % del paper original
time_scaling = 1;
gamma = [0.7, 0.6]; % porcentaje
u0 = [0, 0]; % porcentaje
ti = 0;
Ts = 0;
Hmin = 0.0;

Hmax = 50;
x0 = [40, 40, 40, 40];
voltmax = 10;

% Definir Ti
i = [1, 2, 3, 4];
Ti = Ai(i) ./ ai(i) .* sqrt(2 .* x0(i) ./ g);

% Definir matrices del sistema linealizado en torno al x=1 y u=0
A = [   -1/Ti(1),   0,          Ai(1)/(Ai(3)*Ti(3)),    0; ...
        0,          -1/Ti(2),   0,                      Ai(2)/(Ai(3)*Ti(3)); ...
        0,          0,          -1/Ti(3),               0; ...
        0,          0,          0,                      -1/Ti(4)]
B = [gamma(1)*k(1)/Ai(1),       0; ...
    0,                          gamma(2)*k(2)/Ai(2); ...
    0,                          (1-gamma(2))*k(2)/Ai(3); ...
    (1-gamma(1))*k(1)/Ai(4),    0]
C = [kc,  0,      0, 0; ...
     0,     kc,   0, 0]
D = zeros(2);

% Lazo abierto
ev = eig(A);
if ev < zeros(4,1)
    disp('Estable en lazo abierto');
else
    disp('Inestable en lazo abierto');
end

% Controlabilidad
disp('Sistema linealizado')
CM = ctrb(A, B);
if rank(CM) == size(A, 1)
    disp('The system is controllable');
else
    disp('The system is NOT controllable');
end
disp('---------------------------')

A_star = horzcat([A; C],zeros(6,2));
B_star = vertcat(B,D);

disp('Sistema estrella')
CM = ctrb(A_star, B_star);
if rank(CM) == size(A_star, 1)
    disp('The system is controllable');
else
    disp('The system is NOT controllable');
end

ABC_star = horzcat([A; C],vertcat(B,D));
if rank(ABC_star) == size(ABC_star, 1)
    disp('Full range')
else
    disp('Incomplete range')
end

% LQR
    % Q grande: mayor prioridad en minimizar variables variables, mejora tracking del estado
    % Q chico: menor prioridad en minimizar variables variables, empeora tracking del estado
    % R grande: mayor importancia a control input, resultando en un control suave y conservador 
    % R chico: menor importancia a control input, resultando en un control agresivo y energético
Q = eye(4);
R = eye(1);
N = zeros(4,2);
[K,S,P] = lqr(A,B,Q,R,N);
K

x0 = [40, 40, 40, 40];
ref = [0,0,0,0];
tsim = 5000;
sim("LQR.slx")

