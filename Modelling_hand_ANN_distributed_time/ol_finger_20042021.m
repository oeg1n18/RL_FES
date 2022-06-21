% this file simulates the model descripbed in Anna Soska's PhD chapter 3

clear all, close all, clc
% warning('off')

%% joint lengths (see page 91 of thesis)
l1 = 0.05; l2 = 0.02; l3 = 0.02;

%% Initialisation: set the initial joint angles and angular velocities
load ('settings2.mat','q10','q20','q30','q1dot0','q2dot0','q3dot0');
q1init = q10; clear q10
q2init = q20; clear q20
q3init = q30; clear q30

Ts = 0.01; %% sample time
T = 6; %% length of simulation (seconds)
t_ref = 0:Ts:T;
N = length(t_ref);

%% Linear muscle model:
wn = 2.7; %% the natural frequency of the "linear activation dynamics (LAD)" (see thesis page 38)
%% these dynamics are the system tf(1,[1/wn^2    2/wn   1]); with input FES and output muscle force

global P11 P12 P13 P21 P22 P23 P31 P32 P33
global P41 P42 P43 P51 P52 P53 P61 P62 P63
global P71 P72 P73
global Ts

P11 = 0.0;
P12 = 0.0;
P13 = 0.0;
P21 = 0;
P22 = 0.0;
P23 = 0.0;
P31 = 0.0;
P32 = 0.0;
P33 = 0.0;
P41 = 0.0;
P42 = 0.0;
P43 = 0.0;
P51 = 0.0;
P52 = 0.0;
P53 = 0.0;
P61 = 0.0;
P62 = 0.0;
P63 = 0.0;
P71 = 0.0;
P72 = 0.0;
P73 = 0.0;

%% NOTE: usually muscles are modelled as a Hammerstein system with an "isometric recruitment curve (IRC)"
%% static function that goes before the LAD. However in this model we ignore the IRC for simplicity.
%% This just means that the "stimulation" we apply as input will be very small (less than 1) as is
%% basically corresponds to torque

%% to test the model, let's apply a ramp to some of the 7 muscles
%% values umax > 0.2 make the outputs go extremely large as we are applying a ramp and we only have a simple stiffness form
umax = 0.15; %% peak stimulation of ramp

%% M = [ M_FDP, M_LU, M_UI, M_RI, M_EC, M_ECR, M_ECU]; %% muscle order

i1 = 5; %% STIMULATE EC MUSCLE
%i1 = 7; %% ECU
%i1 = 6; %% ECR

% Define initial control input
ramp = [0:Ts:1 ones(1,N-101)]';

%% construct input for all 7 muscles
u = zeros(N,7);
u(:,i1) = umax*ramp;
disp("input");
disp(size(u));

sim('finger_mus2_20042021');
yout = q.signals.values;

%% record steady-state output
ystore = yout(end,:);
disp("output");
disp(size(ystore));
disp(ystore);

%% plot results in realtime
figure
tic, tc = 0;
while tc < T % 1:1:length(yout(:,1))
    tc = toc;
    i = floor(tc/Ts); i = max(min(i,N),1); %% convert to a sample
    p0 = [0; 0];
    p1 = p0 + l1*[sin(yout(i,1)); cos(yout(i,1))];
    p2 = p1 + l2*[sin(yout(i,1)+yout(i,2)); cos(yout(i,1)+yout(i,2))];
    p3 = p2 + l3*[sin(yout(i,1)+yout(i,2)+yout(i,3)); cos(yout(i,1)+yout(i,2)+yout(i,3))];
    
    plot([p0(1) p1(1) p2(1) p3(1)],[p0(2) p1(2) p2(2) p3(2)],'-ko'); hold off
    xlim([-0.15 0.15]); ylim([-0.15 0.15]); daspect([1 1 1]); grid on;
    pause(0.1)
    
end
% clf

figure
u = u*50; %% add scalar linear recruitment curve so input becomes FES
subplot(211)
plot(q.time,yout(:,1),'k:',q.time,yout(:,2),'k-.',q.time,yout(:,3),'k--'); legend('theta1','theta2','theta3');
xlabel('Time (s)'); ylabel('\theta (rad)'); hold on;
subplot(212)
plot(q.time,u(:,1),'k:',q.time,u(:,2),'k-.',q.time,u(:,3),'k--',q.time,u(:,4),'b',q.time,u(:,5),'r',q.time,u(:,6),'c',q.time,u(:,7),'g'); legend('u1','u2','u3','u4','u5','u6','u7');
xlabel('Time (s)'); ylabel('Stimulation'); hold on;
