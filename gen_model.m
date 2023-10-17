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
model.L_birth = 2;
model.w_birth= zeros(model.L_birth,1);                                %weights of Gaussian birth terms (per duration)
model.m_birth= zeros(model.xdim,model.L_birth);                       %means of Gaussian birth terms 
model.P_birth= zeros(model.xdim,model.xdim,model.L_birth);            %cov of Gaussian birth terms

model.w_birth= [.01; .01];                                            %birth of simulation (example 1)
model.m_birth= [ [250; 250; 0; 0] [-250; -250; 0; 0] ];
model.P_birth(:,:,1)= diag([100, 100, 25, 25]);
model.P_birth(:,:,2)= diag([100, 100, 25, 25]);

model.w_birth_2= [.01; .01];                                          %birth of simulation_2
model.m_birth_2= [ [647; -435; 0; 0] [-250; -250; 0; 0] ];
model.P_birth_2(:,:,1)= diag([100, 100, 25, 25]);
model.P_birth_2(:,:,2)= diag([100, 100, 25, 25]);

model.w_birth_3= .01;                                                 %birth of comparison
model.m_birth_3= [0; 0; 0; 0] ;
model.P_birth_3= diag([20, 20, 10, 10]);

% Detection parameters
model.P_D = 0.99;       % probability of detection in measurements
model.P_MD= 1-model.P_D; % probability of missed detection in measurements

% Clutter parameters
model.lambda_c = 50; % clutter rate
model.range_c= [-1000 1000; -1000 1000];      % uniform clutter region
model.pdf_c= 1/prod(model.range_c(:,2)-model.range_c(:,1)); % uniform clutter density

% Extended Parameters
model.carSize = [4.85; 1.85];
model.truckSize = [15; 3];