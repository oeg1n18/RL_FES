function [qdot] = finger_for_simulink(q)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                   %
% Input variables are the three angles, corresponding ngular       %
% velocities and the applied torques:                               %
% q1, q2, q3, q1dot, q2dot, q3dot,                                  %
% u1, u2, u3                                                        %
% v1,v2 ...v7 - input forces                                              %                    %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%CTF: remember --- if using M(), model has singularity at 0, pi, 2*pi etc...

%% Input
q1 = q(1);
q2 = q(2);
q3 = q(3);
q1dot = q(4);
q2dot = q(5);
q3dot = q(6);
q7 = q(7);
q8 = q(8);
q9 = q(9);
q10 = q(10);
q11 = q(11);
q12 = q(12);
%% inputs provided externally - these are in Nm since we have no IRC
v1 = q(13);
v2 = q(14); %LU
v3 = q(15);
v4 = q(16);
v5 = q(17);
v6 = q(18);
v7 = q(19);

%******************************************
% The M Matrix
%******************************************
% Parameters of the muscles
w1 = 0.3;  %% PAGE 37, eqn (3.37) can't find value
w2 = 0.3;  %% PAGE 37, eqn (3.37) can't find value
w3 = 0.4;  %% PAGE 37, eqn (3.37) can't find value
% Parameters of the tendon model
rTE = 1.88;    %%? not used
y3_FDP = 2.97; %%? not used
d3_FDP = 3.96; %%? not used
d2_FDS = 4.13; %%? not used
y2_FDS = 6.73; %%? not used
d2_FDP = 5.76; %% table PAGE 90   FDP PIP
y2_FDP = 7.5;  %% table PAGE 90   FDP PIP
d1_FDP = 8.32; %% table PAGE 90   FDP
y1_FDP = 8.32; %% table PAGE 90   FDP
bLU = 12.53;   %% table PAGE 90
hLU = -2.17;   %% table PAGE 90
bRB = 2.54;    %% table PAGE 90
hRB = -0.47;   %% table PAGE 90
bUI = 18.76;   %% table PAGE 90
hUI = -8.16;   %% table PAGE 90
bUB = 1.7;     %% table PAGE 90
hUB = 0.57;    %% table PAGE 90
bRI = 5.62;    %% table PAGE 90
hRI = -1.29;   %% table PAGE 90
d1_FDS = 9.56; %%? not used
y1_FDS = 8.14; %%? not used
rEI = 8.82;    %%? not used
rEC1 = 8.3;    %% table PAGE 90
rEC2 = 10;     %%?
rES = 2.92;    %% table PAGE 90
rECR = 9.5;    %%?
rECU = 9.2;    %%?
rFDP= 7.1;     %%?
 
%% PAGE 37, eqn (3.40) %%%%%%%%%%%%%%%
M_FDP = [rFDP;
         d1_FDP + y1_FDP * (sin(q2)-q2)/(2*sin(q2)^2);
         d2_FDP + y2_FDP * (sin(q3)-q3)/(2*sin(q3)^2);
         ];
%% PAGE 37, eqn (3.41) %%%%%%%%%%%%%%%     
M_LU = [0;
        bLU + 2*hLU*q2 - M_FDP(2,1);
        -bRB - 2*hRB*q3 - M_FDP(3,1);
        ];    
%% PAGE 37, eqn (3.42) %%%%%%%%%%%%%%%      
M_UI = [ 0;
         bUI + 2*hUI*q2;
         -bUB - 2*hUB*q3;
                   ];   
%% PAGE 37, eqn (3.38) comapre with M_UI %%%%%%%%%%%%%%%                  
M_RI = [ 0;
        bRI  + 2*hRI*q2;
        0;
                    ]; 
%% PAGE 38, eqn (3.43) %%%%%%%%%%%%%%%    
M_EC = [ -rEC2
          -rEC1;
          -w1*rES - w2*(bUB +2*hUB*q3) - w3*(bRB + 2*hRB*q3);
        ];
%% PAGE 38, eqn (3.46) %%%%%%%%%%%%%%%    
M_ECR = [-rECR; 0; 0;];
%% PAGE 38, eqn (3.47) %%%%%%%%%%%%%%%    
M_ECU = [-rECU; 0; 0;];

%% PAGE 38, eqn (3.39) %%%%%%%%%%%%%%%    
M = [ M_FDP, M_LU, M_UI, M_RI, M_EC, M_ECR, M_ECU];

%THE FORM MUST MATCH WITH THE FORM IN Linearize_finger2nomus.m
if(1)
    u1 = M(1,1)*v1+M(1,2)*v2+M(1,3)*v3+M(1,4)*v4+M(1,5)*v5+M(1,6)*v6+M(1,7)*v7;
    u2 = M(2,1)*v1+M(2,2)*v2+M(2,3)*v3+M(2,4)*v4+M(2,5)*v5+M(2,6)*v6+M(2,7)*v7;
    u3 = M(3,1)*v1+M(3,2)*v2+M(3,3)*v3+M(3,4)*v4+M(3,5)*v5+M(3,6)*v6+M(3,7)*v7;
elseif(0)
    %u1 = q1*v1+q2*v6+q3*v7;
    %u2 = q1*v1+q2*v6+q3*v7;
    %u3 = q1*v1+q2*v6+q3*v7;
else
    u1 = v1;
    u2 = v5;
    u3 = v7;
end

%% initial joint angkes PAGE 90/1
q1init = (2/3)*pi; %rads
q2init = pi/2;    
q3init = pi/3;     

%% Model parameters
% Define Gender (Male or Female)
gender = 'Male';

% Define condition (unimpaired or impaired)
condition = 'Unimpaired';

if (strcmp(gender,'Male')), 
    % Model parameters Male
    m1 = 0.05;          % mass palm in kg       PAGE 90
    m2 = 0.04;          % mass fingers in kg    PAGE 90
    m3 = 0.03;          % mass thumb in kg      PAGE 90
    l1 = 0.05;          % length palm in m      PAGE 90
    l2 = 0.02;          % length fingers in m   PAGE 91
    l3 = 0.02;          % length thumb in m     PAGE 91
    a1 = 0.025;         % length wrist to COM palm in m                  PAGE 91
    a2 = 0.01;          % length fingers MCP joint to COM fingers in m   PAGE 91
    a3 = 0.01;          % length thumb CMP joint to COM thumb in m       PAGE 91
elseif (strcmp(gender,'Female')),
    % Model parameters Female
    m1 = 0.04;          % mass palm in kg
    m2 = 0.02;          % mass fingers in kg
    m3 = 0.02;          % mass thumb in kg
    l1 = 0.05;          % length palm in m
    l2 = 0.02;          % length fingers in m
    l3 = 0.02;          % length thumb in m
    a1 = 0.025;         % length wrist to COM palm in m
    a2 = 0.01;          % length fingers MCP joint to COM fingers in m
    a3 = 0.01;         % length thumb CMP joint to COM thumb in m
else
    display('Gender entered incorrectly')
end

if (strcmp(condition,'Unimpaired')),
    % Model parameters unimpaired subject
    b1 = 0.5; % viscous friction parameter in kg m s^-2   PAGE 91
    b2 = 0.4; % viscous friction parameter in kg m s^-2   PAGE 91
    b3 = 0.3; % viscous friction parameter in kg m s^-2   PAGE 91
elseif (strcmp(condition,'Impaired')),
    % Model parameters impaired subject
    b1 = 0.902; % viscous friction parameter in kg m s^-2
    b2 = 0.902; % viscous friction parameter in kg m s^-2
    b3 = 0.902; % viscous friction parameter in kg m s^-2
else
    display('Condition entered incorrectly')
end

% Other model parameters
J2 = 1e-4;   %% PAGE 91
J4 = 1e-4;   %% PAGE 91    
J6 = 1e-4;   %% PAGE 91
k1 =  0.8;   %% spring constant in Nm/rad   PAGE 91
k2 =  0.8;   %% torsional spring constant in Nm/rad   PAGE 91
k3 =  0.8;   %% torsional spring constant in Nm/rad   PAGE 91
wn = 2.7;

%% Equations of motion, determine output
qdot = [ q(4:6);
         (l2^2*m3^2*cos(q3)^2*a3^2-m2*l2^2*m3*a3^2-J4*m3*a3^2-l2^2*m3*J6-l2^2*m3^2*a3^2-m2*l2^2*J6-J4*J6)/(-J2*m2*l2^2*m3*a3^2-m3*l1^2*J4*J6-m3^2*l1^2*J4*a3^2-m3^2*l1^2*l2^2*J6-m3^3*l1^2*l2^2*a3^2-a1^2*m1*J4*J6-m2*l1^2*J4*J6-m2^2*l1^2*l2^2*J6-J2*J4*m3*a3^2-J2*l2^2*m3*J6-J2*l2^2*m3^2*a3^2-J2*m2*l2^2*J6-J2*J4*J6-2*m3*l1^2*m2*l2^2*J6-2*m3^2*l1^2*m2*l2^2*a3^2-a1^2*m1*J4*m3*a3^2-a1^2*m1*l2^2*m3*J6-a1^2*m1*l2^2*m3^2*a3^2-a1^2*m1*m2*l2^2*J6-a1^2*m1*m2*l2^2*m3*a3^2-m2*l1^2*J4*m3*a3^2-m2^2*l1^2*l2^2*m3*a3^2+m3^3*l1^2*l2^2*cos(q3)^2*a3^2+a1^2*m1*l2^2*m3^2*cos(q3)^2*a3^2+m2*l1^2*l2^2*m3^2*cos(q3)^2*a3^2+J2*l2^2*m3^2*cos(q3)^2*a3^2+m3^3*l1^2*a3^2*cos(q3+q2)^2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*m2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*J4+l2^2*m3^3*l1^2*cos(q2)^2*a3^2+2*l2^2*m3^2*l1^2*cos(q2)^2*m2*a3^2+l2^2*m3^2*l1^2*cos(q2)^2*J6+2*l2^2*m3*l1^2*cos(q2)^2*m2*J6-2*l2^2*m3^3*l1^2*cos(q2)*a3^2*cos(q3+q2)*cos(q3)+l2^2*m2^2*l1^2*cos(q2)^2*m3*a3^2+l2^2*m2^2*l1^2*cos(q2)^2*J6-2*l2^2*m2*l1^2*cos(q2)*m3^2*a3^2*cos(q3+q2)*cos(q3))*(-k1*(q1-q1init)+l1*(2*m3*q1dot*a3*sin(q3+q2)+2*m2*q1dot*l2*sin(q2)+2*m3*q1dot*l2*sin(q2)+m3*l2*q2dot*sin(q2)+m3*a3*q3dot*sin(q3+q2)+m2*l2*q2dot*sin(q2)+m3*q2dot*a3*sin(q3+q2))*q2dot+m3*a3*(2*q1dot*l1*sin(q3+q2)+2*q1dot*l2*sin(q3)+2*l2*q2dot*sin(q3)+q3dot*l1*sin(q3+q2)+q3dot*l2*sin(q3)+q2dot*l1*sin(q3+q2))*q3dot+q7-b1*q1dot)-(-J4*J6-l2^2*m3^2*a3^2-J4*m3*a3^2-l2^2*m3*J6-m2*l2^2*J6-m2*l2^2*m3*a3^2+l2^2*m3^2*cos(q3)^2*a3^2-l2*m3^2*l1*cos(q2)*a3^2-l2*m2*l1*cos(q2)*m3*a3^2-l2*m3*l1*cos(q2)*J6-l2*m2*l1*cos(q2)*J6+m3^2*l1*a3^2*cos(q3+q2)*l2*cos(q3))/(-J2*m2*l2^2*m3*a3^2-m3*l1^2*J4*J6-m3^2*l1^2*J4*a3^2-m3^2*l1^2*l2^2*J6-m3^3*l1^2*l2^2*a3^2-a1^2*m1*J4*J6-m2*l1^2*J4*J6-m2^2*l1^2*l2^2*J6-J2*J4*m3*a3^2-J2*l2^2*m3*J6-J2*l2^2*m3^2*a3^2-J2*m2*l2^2*J6-J2*J4*J6-2*m3*l1^2*m2*l2^2*J6-2*m3^2*l1^2*m2*l2^2*a3^2-a1^2*m1*J4*m3*a3^2-a1^2*m1*l2^2*m3*J6-a1^2*m1*l2^2*m3^2*a3^2-a1^2*m1*m2*l2^2*J6-a1^2*m1*m2*l2^2*m3*a3^2-m2*l1^2*J4*m3*a3^2-m2^2*l1^2*l2^2*m3*a3^2+m3^3*l1^2*l2^2*cos(q3)^2*a3^2+a1^2*m1*l2^2*m3^2*cos(q3)^2*a3^2+m2*l1^2*l2^2*m3^2*cos(q3)^2*a3^2+J2*l2^2*m3^2*cos(q3)^2*a3^2+m3^3*l1^2*a3^2*cos(q3+q2)^2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*m2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*J4+l2^2*m3^3*l1^2*cos(q2)^2*a3^2+2*l2^2*m3^2*l1^2*cos(q2)^2*m2*a3^2+l2^2*m3^2*l1^2*cos(q2)^2*J6+2*l2^2*m3*l1^2*cos(q2)^2*m2*J6-2*l2^2*m3^3*l1^2*cos(q2)*a3^2*cos(q3+q2)*cos(q3)+l2^2*m2^2*l1^2*cos(q2)^2*m3*a3^2+l2^2*m2^2*l1^2*cos(q2)^2*J6-2*l2^2*m2*l1^2*cos(q2)*m3^2*a3^2*cos(q3+q2)*cos(q3))*(-k2*(q2-q2init)-m3*q1dot*l1*l2*q2dot*sin(q2)-m3*q1dot^2*l1*a3*sin(q3+q2)-m3*q1dot*l1*a3*q3dot*sin(q3+q2)-m2*q1dot^2*l1*l2*sin(q2)-m2*q1dot*l1*l2*q2dot*sin(q2)-m3*q1dot^2*l2*l1*sin(q2)-m3*q1dot*l1*a3*sin(q3+q2)*q2dot+q1dot*l1*(m3*l2*sin(q2)+m3*a3*sin(q3+q2)+m2*l2*sin(q2))*q2dot+m3*a3*(q1dot*l1*sin(q3+q2)+2*q1dot*l2*sin(q3)+q3dot*l2*sin(q3)+2*l2*q2dot*sin(q3))*q3dot+q9-b2*q2dot)+l1*(-l2^2*m3^2*cos(q2)*cos(q3)*a3-l2^2*m2*cos(q2)*m3*cos(q3)*a3+m3^2*a3*cos(q3+q2)*l2^2+m3*a3*cos(q3+q2)*m2*l2^2+m3*a3*cos(q3+q2)*J4-l2*m3^2*cos(q2)*a3^2-l2*m2*cos(q2)*m3*a3^2-l2*m3*cos(q2)*J6-l2*m2*cos(q2)*J6+m3^2*a3^2*cos(q3+q2)*l2*cos(q3))/(-J2*m2*l2^2*m3*a3^2-m3*l1^2*J4*J6-m3^2*l1^2*J4*a3^2-m3^2*l1^2*l2^2*J6-m3^3*l1^2*l2^2*a3^2-a1^2*m1*J4*J6-m2*l1^2*J4*J6-m2^2*l1^2*l2^2*J6-J2*J4*m3*a3^2-J2*l2^2*m3*J6-J2*l2^2*m3^2*a3^2-J2*m2*l2^2*J6-J2*J4*J6-2*m3*l1^2*m2*l2^2*J6-2*m3^2*l1^2*m2*l2^2*a3^2-a1^2*m1*J4*m3*a3^2-a1^2*m1*l2^2*m3*J6-a1^2*m1*l2^2*m3^2*a3^2-a1^2*m1*m2*l2^2*J6-a1^2*m1*m2*l2^2*m3*a3^2-m2*l1^2*J4*m3*a3^2-m2^2*l1^2*l2^2*m3*a3^2+m3^3*l1^2*l2^2*cos(q3)^2*a3^2+a1^2*m1*l2^2*m3^2*cos(q3)^2*a3^2+m2*l1^2*l2^2*m3^2*cos(q3)^2*a3^2+J2*l2^2*m3^2*cos(q3)^2*a3^2+m3^3*l1^2*a3^2*cos(q3+q2)^2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*m2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*J4+l2^2*m3^3*l1^2*cos(q2)^2*a3^2+2*l2^2*m3^2*l1^2*cos(q2)^2*m2*a3^2+l2^2*m3^2*l1^2*cos(q2)^2*J6+2*l2^2*m3*l1^2*cos(q2)^2*m2*J6-2*l2^2*m3^3*l1^2*cos(q2)*a3^2*cos(q3+q2)*cos(q3)+l2^2*m2^2*l1^2*cos(q2)^2*m3*a3^2+l2^2*m2^2*l1^2*cos(q2)^2*J6-2*l2^2*m2*l1^2*cos(q2)*m3^2*a3^2*cos(q3+q2)*cos(q3))*(-k3*(q3-q3init)-m3*q1dot*l2*a3*q3dot*sin(q3)-m3*l2*q2dot^2*a3*sin(q3)-m3*l2*q2dot*a3*q3dot*sin(q3)-m3*q1dot^2*l2*a3*sin(q3)-m3*q1dot^2*l1*a3*sin(q3+q2)-m3*q1dot*l1*a3*q3dot*sin(q3+q2)-2*m3*q1dot*a3*l2*q2dot*sin(q3)+m3*a3*(q1dot*l1*sin(q3+q2)+q1dot*l2*sin(q3)+l2*q2dot*sin(q3))*q3dot+q11-b3*q3dot);
         -(-J4*J6-l2^2*m3^2*a3^2-J4*m3*a3^2-l2^2*m3*J6-m2*l2^2*J6-m2*l2^2*m3*a3^2+l2^2*m3^2*cos(q3)^2*a3^2-l2*m3^2*l1*cos(q2)*a3^2-l2*m2*l1*cos(q2)*m3*a3^2-l2*m3*l1*cos(q2)*J6-l2*m2*l1*cos(q2)*J6+m3^2*l1*a3^2*cos(q3+q2)*l2*cos(q3))/(-J2*m2*l2^2*m3*a3^2-m3*l1^2*J4*J6-m3^2*l1^2*J4*a3^2-m3^2*l1^2*l2^2*J6-m3^3*l1^2*l2^2*a3^2-a1^2*m1*J4*J6-m2*l1^2*J4*J6-m2^2*l1^2*l2^2*J6-J2*J4*m3*a3^2-J2*l2^2*m3*J6-J2*l2^2*m3^2*a3^2-J2*m2*l2^2*J6-J2*J4*J6-2*m3*l1^2*m2*l2^2*J6-2*m3^2*l1^2*m2*l2^2*a3^2-a1^2*m1*J4*m3*a3^2-a1^2*m1*l2^2*m3*J6-a1^2*m1*l2^2*m3^2*a3^2-a1^2*m1*m2*l2^2*J6-a1^2*m1*m2*l2^2*m3*a3^2-m2*l1^2*J4*m3*a3^2-m2^2*l1^2*l2^2*m3*a3^2+m3^3*l1^2*l2^2*cos(q3)^2*a3^2+a1^2*m1*l2^2*m3^2*cos(q3)^2*a3^2+m2*l1^2*l2^2*m3^2*cos(q3)^2*a3^2+J2*l2^2*m3^2*cos(q3)^2*a3^2+m3^3*l1^2*a3^2*cos(q3+q2)^2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*m2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*J4+l2^2*m3^3*l1^2*cos(q2)^2*a3^2+2*l2^2*m3^2*l1^2*cos(q2)^2*m2*a3^2+l2^2*m3^2*l1^2*cos(q2)^2*J6+2*l2^2*m3*l1^2*cos(q2)^2*m2*J6-2*l2^2*m3^3*l1^2*cos(q2)*a3^2*cos(q3+q2)*cos(q3)+l2^2*m2^2*l1^2*cos(q2)^2*m3*a3^2+l2^2*m2^2*l1^2*cos(q2)^2*J6-2*l2^2*m2*l1^2*cos(q2)*m3^2*a3^2*cos(q3+q2)*cos(q3))*(-k1*(q1-q1init)+l1*(2*m3*q1dot*a3*sin(q3+q2)+2*m2*q1dot*l2*sin(q2)+2*m3*q1dot*l2*sin(q2)+m3*l2*q2dot*sin(q2)+m3*a3*q3dot*sin(q3+q2)+m2*l2*q2dot*sin(q2)+m3*q2dot*a3*sin(q3+q2))*q2dot+m3*a3*(2*q1dot*l1*sin(q3+q2)+2*q1dot*l2*sin(q3)+2*l2*q2dot*sin(q3)+q3dot*l1*sin(q3+q2)+q3dot*l2*sin(q3)+q2dot*l1*sin(q3+q2))*q3dot+q7-b1*q1dot)+(-J4*J6-l2^2*m3^2*a3^2-J4*m3*a3^2-l2^2*m3*J6-m2*l2^2*J6-m2*l2^2*m3*a3^2+l2^2*m3^2*cos(q3)^2*a3^2-m3^2*l1^2*a3^2-m3*l1^2*J6-J2*m3*a3^2-a1^2*m1*J6-m2*l1^2*J6-2*l2*m3^2*l1*cos(q2)*a3^2-2*l2*m2*l1*cos(q2)*m3*a3^2-2*l2*m3*l1*cos(q2)*J6-2*l2*m2*l1*cos(q2)*J6+2*m3^2*l1*a3^2*cos(q3+q2)*l2*cos(q3)-a1^2*m1*m3*a3^2-m2*l1^2*m3*a3^2+m3^2*l1^2*a3^2*cos(q3+q2)^2-J2*J6)/(-J2*m2*l2^2*m3*a3^2-m3*l1^2*J4*J6-m3^2*l1^2*J4*a3^2-m3^2*l1^2*l2^2*J6-m3^3*l1^2*l2^2*a3^2-a1^2*m1*J4*J6-m2*l1^2*J4*J6-m2^2*l1^2*l2^2*J6-J2*J4*m3*a3^2-J2*l2^2*m3*J6-J2*l2^2*m3^2*a3^2-J2*m2*l2^2*J6-J2*J4*J6-2*m3*l1^2*m2*l2^2*J6-2*m3^2*l1^2*m2*l2^2*a3^2-a1^2*m1*J4*m3*a3^2-a1^2*m1*l2^2*m3*J6-a1^2*m1*l2^2*m3^2*a3^2-a1^2*m1*m2*l2^2*J6-a1^2*m1*m2*l2^2*m3*a3^2-m2*l1^2*J4*m3*a3^2-m2^2*l1^2*l2^2*m3*a3^2+m3^3*l1^2*l2^2*cos(q3)^2*a3^2+a1^2*m1*l2^2*m3^2*cos(q3)^2*a3^2+m2*l1^2*l2^2*m3^2*cos(q3)^2*a3^2+J2*l2^2*m3^2*cos(q3)^2*a3^2+m3^3*l1^2*a3^2*cos(q3+q2)^2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*m2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*J4+l2^2*m3^3*l1^2*cos(q2)^2*a3^2+2*l2^2*m3^2*l1^2*cos(q2)^2*m2*a3^2+l2^2*m3^2*l1^2*cos(q2)^2*J6+2*l2^2*m3*l1^2*cos(q2)^2*m2*J6-2*l2^2*m3^3*l1^2*cos(q2)*a3^2*cos(q3+q2)*cos(q3)+l2^2*m2^2*l1^2*cos(q2)^2*m3*a3^2+l2^2*m2^2*l1^2*cos(q2)^2*J6-2*l2^2*m2*l1^2*cos(q2)*m3^2*a3^2*cos(q3+q2)*cos(q3))*(-k2*(q2-q2init)-m3*q1dot*l1*l2*q2dot*sin(q2)-m3*q1dot^2*l1*a3*sin(q3+q2)-m3*q1dot*l1*a3*q3dot*sin(q3+q2)-m2*q1dot^2*l1*l2*sin(q2)-m2*q1dot*l1*l2*q2dot*sin(q2)-m3*q1dot^2*l2*l1*sin(q2)-m3*q1dot*l1*a3*sin(q3+q2)*q2dot+q1dot*l1*(m3*l2*sin(q2)+m3*a3*sin(q3+q2)+m2*l2*sin(q2))*q2dot+m3*a3*(q1dot*l1*sin(q3+q2)+2*q1dot*l2*sin(q3)+q3dot*l2*sin(q3)+2*l2*q2dot*sin(q3))*q3dot+q9-b2*q2dot)-(-l2*m2*l1*cos(q2)*J6-l2^2*m3^2*l1*cos(q2)*cos(q3)*a3-l2^2*m2*l1*cos(q2)*m3*cos(q3)*a3+m3^2*l1*a3*cos(q3+q2)*l2^2+m3*l1*a3*cos(q3+q2)*m2*l2^2+m3*l1*a3*cos(q3+q2)*J4-m3^2*l1^2*a3^2-m3*l1^2*J6-J2*m3*a3^2-a1^2*m1*J6-m2*l1^2*J6-l2*m3^2*l1*cos(q2)*a3^2-l2*m2*l1*cos(q2)*m3*a3^2-l2*m3*l1*cos(q2)*J6+m3^2*l1*a3^2*cos(q3+q2)*l2*cos(q3)-a1^2*m1*m3*a3^2-m2*l1^2*m3*a3^2+m3^2*l1^2*a3^2*cos(q3+q2)^2-J2*J6-m3^2*l1^2*l2*cos(q3)*a3-a1^2*m1*l2*m3*cos(q3)*a3-m2*l1^2*l2*m3*cos(q3)*a3-J2*l2*m3*cos(q3)*a3+l2*m3^2*l1^2*cos(q2)*a3*cos(q3+q2)+l2*m2*l1^2*cos(q2)*m3*a3*cos(q3+q2))/(-J2*m2*l2^2*m3*a3^2-m3*l1^2*J4*J6-m3^2*l1^2*J4*a3^2-m3^2*l1^2*l2^2*J6-m3^3*l1^2*l2^2*a3^2-a1^2*m1*J4*J6-m2*l1^2*J4*J6-m2^2*l1^2*l2^2*J6-J2*J4*m3*a3^2-J2*l2^2*m3*J6-J2*l2^2*m3^2*a3^2-J2*m2*l2^2*J6-J2*J4*J6-2*m3*l1^2*m2*l2^2*J6-2*m3^2*l1^2*m2*l2^2*a3^2-a1^2*m1*J4*m3*a3^2-a1^2*m1*l2^2*m3*J6-a1^2*m1*l2^2*m3^2*a3^2-a1^2*m1*m2*l2^2*J6-a1^2*m1*m2*l2^2*m3*a3^2-m2*l1^2*J4*m3*a3^2-m2^2*l1^2*l2^2*m3*a3^2+m3^3*l1^2*l2^2*cos(q3)^2*a3^2+a1^2*m1*l2^2*m3^2*cos(q3)^2*a3^2+m2*l1^2*l2^2*m3^2*cos(q3)^2*a3^2+J2*l2^2*m3^2*cos(q3)^2*a3^2+m3^3*l1^2*a3^2*cos(q3+q2)^2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*m2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*J4+l2^2*m3^3*l1^2*cos(q2)^2*a3^2+2*l2^2*m3^2*l1^2*cos(q2)^2*m2*a3^2+l2^2*m3^2*l1^2*cos(q2)^2*J6+2*l2^2*m3*l1^2*cos(q2)^2*m2*J6-2*l2^2*m3^3*l1^2*cos(q2)*a3^2*cos(q3+q2)*cos(q3)+l2^2*m2^2*l1^2*cos(q2)^2*m3*a3^2+l2^2*m2^2*l1^2*cos(q2)^2*J6-2*l2^2*m2*l1^2*cos(q2)*m3^2*a3^2*cos(q3+q2)*cos(q3))*(-k3*(q3-q3init)-m3*q1dot*l2*a3*q3dot*sin(q3)-m3*l2*q2dot^2*a3*sin(q3)-m3*l2*q2dot*a3*q3dot*sin(q3)-m3*q1dot^2*l2*a3*sin(q3)-m3*q1dot^2*l1*a3*sin(q3+q2)-m3*q1dot*l1*a3*q3dot*sin(q3+q2)-2*m3*q1dot*a3*l2*q2dot*sin(q3)+m3*a3*(q1dot*l1*sin(q3+q2)+q1dot*l2*sin(q3)+l2*q2dot*sin(q3))*q3dot+q11-b3*q3dot);
         l1*(-l2^2*m3^2*cos(q2)*cos(q3)*a3-l2^2*m2*cos(q2)*m3*cos(q3)*a3+m3^2*a3*cos(q3+q2)*l2^2+m3*a3*cos(q3+q2)*m2*l2^2+m3*a3*cos(q3+q2)*J4-l2*m3^2*cos(q2)*a3^2-l2*m2*cos(q2)*m3*a3^2-l2*m3*cos(q2)*J6-l2*m2*cos(q2)*J6+m3^2*a3^2*cos(q3+q2)*l2*cos(q3))/(-J2*m2*l2^2*m3*a3^2-m3*l1^2*J4*J6-m3^2*l1^2*J4*a3^2-m3^2*l1^2*l2^2*J6-m3^3*l1^2*l2^2*a3^2-a1^2*m1*J4*J6-m2*l1^2*J4*J6-m2^2*l1^2*l2^2*J6-J2*J4*m3*a3^2-J2*l2^2*m3*J6-J2*l2^2*m3^2*a3^2-J2*m2*l2^2*J6-J2*J4*J6-2*m3*l1^2*m2*l2^2*J6-2*m3^2*l1^2*m2*l2^2*a3^2-a1^2*m1*J4*m3*a3^2-a1^2*m1*l2^2*m3*J6-a1^2*m1*l2^2*m3^2*a3^2-a1^2*m1*m2*l2^2*J6-a1^2*m1*m2*l2^2*m3*a3^2-m2*l1^2*J4*m3*a3^2-m2^2*l1^2*l2^2*m3*a3^2+m3^3*l1^2*l2^2*cos(q3)^2*a3^2+a1^2*m1*l2^2*m3^2*cos(q3)^2*a3^2+m2*l1^2*l2^2*m3^2*cos(q3)^2*a3^2+J2*l2^2*m3^2*cos(q3)^2*a3^2+m3^3*l1^2*a3^2*cos(q3+q2)^2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*m2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*J4+l2^2*m3^3*l1^2*cos(q2)^2*a3^2+2*l2^2*m3^2*l1^2*cos(q2)^2*m2*a3^2+l2^2*m3^2*l1^2*cos(q2)^2*J6+2*l2^2*m3*l1^2*cos(q2)^2*m2*J6-2*l2^2*m3^3*l1^2*cos(q2)*a3^2*cos(q3+q2)*cos(q3)+l2^2*m2^2*l1^2*cos(q2)^2*m3*a3^2+l2^2*m2^2*l1^2*cos(q2)^2*J6-2*l2^2*m2*l1^2*cos(q2)*m3^2*a3^2*cos(q3+q2)*cos(q3))*(-k1*(q1-q1init)+l1*(2*m3*q1dot*a3*sin(q3+q2)+2*m2*q1dot*l2*sin(q2)+2*m3*q1dot*l2*sin(q2)+m3*l2*q2dot*sin(q2)+m3*a3*q3dot*sin(q3+q2)+m2*l2*q2dot*sin(q2)+m3*q2dot*a3*sin(q3+q2))*q2dot+m3*a3*(2*q1dot*l1*sin(q3+q2)+2*q1dot*l2*sin(q3)+2*l2*q2dot*sin(q3)+q3dot*l1*sin(q3+q2)+q3dot*l2*sin(q3)+q2dot*l1*sin(q3+q2))*q3dot+q7-b1*q1dot)-(-l2*m2*l1*cos(q2)*J6-l2^2*m3^2*l1*cos(q2)*cos(q3)*a3-l2^2*m2*l1*cos(q2)*m3*cos(q3)*a3+m3^2*l1*a3*cos(q3+q2)*l2^2+m3*l1*a3*cos(q3+q2)*m2*l2^2+m3*l1*a3*cos(q3+q2)*J4-m3^2*l1^2*a3^2-m3*l1^2*J6-J2*m3*a3^2-a1^2*m1*J6-m2*l1^2*J6-l2*m3^2*l1*cos(q2)*a3^2-l2*m2*l1*cos(q2)*m3*a3^2-l2*m3*l1*cos(q2)*J6+m3^2*l1*a3^2*cos(q3+q2)*l2*cos(q3)-a1^2*m1*m3*a3^2-m2*l1^2*m3*a3^2+m3^2*l1^2*a3^2*cos(q3+q2)^2-J2*J6-m3^2*l1^2*l2*cos(q3)*a3-a1^2*m1*l2*m3*cos(q3)*a3-m2*l1^2*l2*m3*cos(q3)*a3-J2*l2*m3*cos(q3)*a3+l2*m3^2*l1^2*cos(q2)*a3*cos(q3+q2)+l2*m2*l1^2*cos(q2)*m3*a3*cos(q3+q2))/(-J2*m2*l2^2*m3*a3^2-m3*l1^2*J4*J6-m3^2*l1^2*J4*a3^2-m3^2*l1^2*l2^2*J6-m3^3*l1^2*l2^2*a3^2-a1^2*m1*J4*J6-m2*l1^2*J4*J6-m2^2*l1^2*l2^2*J6-J2*J4*m3*a3^2-J2*l2^2*m3*J6-J2*l2^2*m3^2*a3^2-J2*m2*l2^2*J6-J2*J4*J6-2*m3*l1^2*m2*l2^2*J6-2*m3^2*l1^2*m2*l2^2*a3^2-a1^2*m1*J4*m3*a3^2-a1^2*m1*l2^2*m3*J6-a1^2*m1*l2^2*m3^2*a3^2-a1^2*m1*m2*l2^2*J6-a1^2*m1*m2*l2^2*m3*a3^2-m2*l1^2*J4*m3*a3^2-m2^2*l1^2*l2^2*m3*a3^2+m3^3*l1^2*l2^2*cos(q3)^2*a3^2+a1^2*m1*l2^2*m3^2*cos(q3)^2*a3^2+m2*l1^2*l2^2*m3^2*cos(q3)^2*a3^2+J2*l2^2*m3^2*cos(q3)^2*a3^2+m3^3*l1^2*a3^2*cos(q3+q2)^2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*m2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*J4+l2^2*m3^3*l1^2*cos(q2)^2*a3^2+2*l2^2*m3^2*l1^2*cos(q2)^2*m2*a3^2+l2^2*m3^2*l1^2*cos(q2)^2*J6+2*l2^2*m3*l1^2*cos(q2)^2*m2*J6-2*l2^2*m3^3*l1^2*cos(q2)*a3^2*cos(q3+q2)*cos(q3)+l2^2*m2^2*l1^2*cos(q2)^2*m3*a3^2+l2^2*m2^2*l1^2*cos(q2)^2*J6-2*l2^2*m2*l1^2*cos(q2)*m3^2*a3^2*cos(q3+q2)*cos(q3))*(-k2*(q2-q2init)-m3*q1dot*l1*l2*q2dot*sin(q2)-m3*q1dot^2*l1*a3*sin(q3+q2)-m3*q1dot*l1*a3*q3dot*sin(q3+q2)-m2*q1dot^2*l1*l2*sin(q2)-m2*q1dot*l1*l2*q2dot*sin(q2)-m3*q1dot^2*l2*l1*sin(q2)-m3*q1dot*l1*a3*sin(q3+q2)*q2dot+q1dot*l1*(m3*l2*sin(q2)+m3*a3*sin(q3+q2)+m2*l2*sin(q2))*q2dot+m3*a3*(q1dot*l1*sin(q3+q2)+2*q1dot*l2*sin(q3)+q3dot*l2*sin(q3)+2*l2*q2dot*sin(q3))*q3dot+q9-b2*q2dot)+(-m3*l1^2*J4-m2^2*l1^2*l2^2-m3^2*l1^2*l2^2-m2*l1^2*J4-a1^2*m1*J4-J2*m2*l2^2-J2*l2^2*m3-2*m3*l1^2*m2*l2^2-a1^2*m1*m2*l2^2-a1^2*m1*l2^2*m3+l2^2*m2^2*l1^2*cos(q2)^2+l2^2*m3^2*l1^2*cos(q2)^2-J2*J4+2*l2^2*m3*l1^2*cos(q2)^2*m2-m3^2*l1^2*a3^2-m3*l1^2*J6-J2*m3*a3^2-a1^2*m1*J6-m2*l1^2*J6-a1^2*m1*m3*a3^2-m2*l1^2*m3*a3^2+m3^2*l1^2*a3^2*cos(q3+q2)^2-J2*J6-2*m3^2*l1^2*l2*cos(q3)*a3-2*a1^2*m1*l2*m3*cos(q3)*a3-2*m2*l1^2*l2*m3*cos(q3)*a3-2*J2*l2*m3*cos(q3)*a3+2*l2*m3^2*l1^2*cos(q2)*a3*cos(q3+q2)+2*l2*m2*l1^2*cos(q2)*m3*a3*cos(q3+q2))/(-J2*m2*l2^2*m3*a3^2-m3*l1^2*J4*J6-m3^2*l1^2*J4*a3^2-m3^2*l1^2*l2^2*J6-m3^3*l1^2*l2^2*a3^2-a1^2*m1*J4*J6-m2*l1^2*J4*J6-m2^2*l1^2*l2^2*J6-J2*J4*m3*a3^2-J2*l2^2*m3*J6-J2*l2^2*m3^2*a3^2-J2*m2*l2^2*J6-J2*J4*J6-2*m3*l1^2*m2*l2^2*J6-2*m3^2*l1^2*m2*l2^2*a3^2-a1^2*m1*J4*m3*a3^2-a1^2*m1*l2^2*m3*J6-a1^2*m1*l2^2*m3^2*a3^2-a1^2*m1*m2*l2^2*J6-a1^2*m1*m2*l2^2*m3*a3^2-m2*l1^2*J4*m3*a3^2-m2^2*l1^2*l2^2*m3*a3^2+m3^3*l1^2*l2^2*cos(q3)^2*a3^2+a1^2*m1*l2^2*m3^2*cos(q3)^2*a3^2+m2*l1^2*l2^2*m3^2*cos(q3)^2*a3^2+J2*l2^2*m3^2*cos(q3)^2*a3^2+m3^3*l1^2*a3^2*cos(q3+q2)^2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*m2*l2^2+m3^2*l1^2*a3^2*cos(q3+q2)^2*J4+l2^2*m3^3*l1^2*cos(q2)^2*a3^2+2*l2^2*m3^2*l1^2*cos(q2)^2*m2*a3^2+l2^2*m3^2*l1^2*cos(q2)^2*J6+2*l2^2*m3*l1^2*cos(q2)^2*m2*J6-2*l2^2*m3^3*l1^2*cos(q2)*a3^2*cos(q3+q2)*cos(q3)+l2^2*m2^2*l1^2*cos(q2)^2*m3*a3^2+l2^2*m2^2*l1^2*cos(q2)^2*J6-2*l2^2*m2*l1^2*cos(q2)*m3^2*a3^2*cos(q3+q2)*cos(q3))*(-k3*(q3-q3init)-m3*q1dot*l2*a3*q3dot*sin(q3)-m3*l2*q2dot^2*a3*sin(q3)-m3*l2*q2dot*a3*q3dot*sin(q3)-m3*q1dot^2*l2*a3*sin(q3)-m3*q1dot^2*l1*a3*sin(q3+q2)-m3*q1dot*l1*a3*q3dot*sin(q3+q2)-2*m3*q1dot*a3*l2*q2dot*sin(q3)+m3*a3*(q1dot*l1*sin(q3+q2)+q1dot*l2*sin(q3)+l2*q2dot*sin(q3))*q3dot+q11-b3*q3dot);
         q8;  %q7 replaces u1 above
         -2*wn*q8  - (wn^2)*q7  + (wn^2)*u1;
         q10; %q9 replaces u2 above
         -2*wn*q10 - (wn^2)*q9  + (wn^2)*u2;
         q12; %q11 replaces u3 above
         -2*wn*q12 - (wn^2)*q11 + (wn^2)*u3 ];
     
   
