function [input, output] = hand_test(channel1, channel2, amplitude1, amplitude2, Tset)
%UNTITLED2 Summary of this function goes here
%% joint lengths (see page 91 of thesis)
options = simset('SrcWorkspace','current');

l1 = 0.05; l2 = 0.02; l3 = 0.02;

    %% Initialisation: set the initial joint angles and angular velocities
    load ('settings2.mat','q10','q20','q30','q1dot0','q2dot0','q3dot0');
    q1init = q10; clear q10
    q2init = q20; clear q20
    q3init = q30; clear q30

    Ts = 0.02; %% sample time
    T = 5; %% length of simulation (seconds)
    t_ref = 0:Ts:T;
    N = length(t_ref);
    %% Linear muscle model:
    wn = 2.7; %% the natural frequency of the "linear activation dynamics (LAD)" (see thesis page 38)
    %% these dynamics are the system tf(1,[1/wn^2    2/wn   1]); with input FES and output muscle force

    global P11 P12 P13 P21 P22 P23 P31 P32 P33
    global P41 P42 P43 P51 P52 P53 P61 P62 P63
    global P71 P72 P73
    global Ts
    Ts = 0.02;

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
    Tmax =
        
    
    if amplitude1 == 0
        umax1 = 0.02; %% peak stimulation of ramp
    elseif amplitude1 == 1
        umax1 = 0.04;
    elseif amplitude1 == 2
        umax1 = 0.06;
    elseif amplitude1 == 3
        umax1 = 0.08;
    elseif amplitude1 == 4
        umax1 = 0.10;
    elseif amplitude1 == 5
        umax1 = 0.12; 
    elseif amplitude1 == 6
        umax1 = 0.14;%% peak stimulation of ramp
    end
    
    if amplitude2 == 0
        umax2 = 0.02; %% peak stimulation of ramp
    elseif amplitude2 == 1
        umax2 = 0.04;
    elseif amplitude2 == 2
        umax2 = 0.06;
    elseif amplitude2 == 3
        umax2 = 0.08;
    elseif amplitude2 == 4
        umax2 = 0.10;
    elseif amplitude2 == 5
        umax2 = 0.12; 
    elseif amplitude2 == 6
        umax2 = 0.14;%% peak stimulation of ramp
    end
   

    %% M = [ M_FDP, M_LU, M_UI, M_RI, M_EC, M_ECR, M_ECU]; %% muscle order

            
     %% STIMULATE EC MUSCLE

    % Define initial control input
    s = size(0:(Ts/Tmax):1);
    ramp = [0:(Ts/Tmax):1 ones(1,N - s(2))]';
    %% construct input for all 7 muscles

    u = zeros(N,7);

    
    if channel1 == 0
        u(:,channel2) = umax2*ramp;
    elseif channel2 == 0
        u(:,channel1) = umax1*ramp;
    else  
        u(:,channel1) = umax1*ramp;
        u(:,channel2) = umax2*ramp;
    end
    
     
        
    sim('finger_mus2_20042021',[],options);
    yout = q.signals.values;

    %% record steady-state output
    input = transpose(u);
    output = transpose(yout);
end

