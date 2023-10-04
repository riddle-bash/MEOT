function model = gen_model
% Transition model
model.xdim = 4;      % x dimension
model.dt = 1;       % sampling period
model.F = [1 0 model.dt 0;
           0 1 0 model.dt;
           0 0 1 0;
           0 0 0 1;
           ];
model.P_S = 0.99;     % Surviving Probability
% Measurement model
model.zdim = 2;     % z dimension
model.H = [1 0 0 0;
           0 1 0 0;
          ]; 
% Transition Noise
Trans_noise_mag = 15;
%upx^2 = 3, upy^2 = 2.5, uvx^2 = 2, upy^2 = 1
model.Q = [3 0 0 0;
           0 2.5 0 0;
           0 0 2 0;
           0 0 0 1;
          ];
model.Q = model.Q * Trans_noise_mag;
% Measurement Noise
Meas_noise_mag = 5;
%upx^2 = .5, upy^2 = .65, uvx^2 = .4, uvy^2 = .35
model.R = [.5 0;
           0 .65;
          ];
model.R = model.R * Meas_noise_mag;

% Birth parameters
model.L_birth = 1;
model.w_birth= zeros(model.L_birth,1);                                %weights of Gaussian birth terms (per duration)
model.m_birth= zeros(model.xdim,model.L_birth);                       %means of Gaussian birth terms 
model.P_birth= zeros(model.xdim,model.xdim,model.L_birth);            %cov of Gaussian birth terms

model.w_birth(1)= 3/100;                                              %birth term 1
model.m_birth(:,1)= [ 0; 0; 0; 0 ];
model.B_birth(:,:,1)= diag([100 100 100 100]).^2;

% Detection parameters
model.P_D = 0.99;       % probability of detection in measurements
model.P_MD= 1-model.P_D; % probability of missed detection in measurements

% Clutter parameters
model.lambda_c = 5; % clutter rate
model.range_c= [-100 100; -100 100];      % uniform clutter region
model.pdf_c= 1/prod(model.range_c(:,2)-model.range_c(:,1)); % uniform clutter density