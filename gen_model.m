function model = gen_model(m)
% Initial states
model.duration = 20;

% Extended Parameters
model.carSize = [4.85; 1.85];
model.truckSize = [15; 3];

switch m
    case 1
        model.gt1(:,1) = [0; 10; 20; 20];
        model.gt1_shape = [56.3/180 * pi; 1 * model.carSize];

        model.gt2(:,1) = [10; 450; 20; -20];
        model.gt2_shape = [56.3/180 * pi; 1 * model.carSize];
    case 2
        model.gt1(:,1) = [0; 0; 10; 20];
        model.gt1_shape = [56.3/180 * pi; 1 * model.carSize];

        model.gt2(:,1) = [500; 500; -10; -20];
        model.gt2_shape = [56.3/180 * pi; 1 * model.carSize];
end

% Transition model
model.xdim = 4;     % x dimension
model.dt = 1;       % sampling period
model.F = [1 0 model.dt 0;
           0 1 0 model.dt;
           0 0 1 0;
           0 0 0 1;
           ];
model.F2 = [1 0 model.dt 0;
           0 1 0 model.dt;
           0 0 -1 0;
           0 0 0 -1;
           ];
model.P_S = 0.99;     % Surviving Probability

% Measurement model
model.lambda_z = 5;
model.zdim = 2;     % z dimension
model.H = [1 0 0 0;
           0 1 0 0;
          ]; 
% Transition Noise
Trans_noise_mag = 2;
%upx^2 = 3, upy^2 = 2.5, uvx^2 = 2, upy^2 = 1
model.Q = [3 0 0 0;
           0 2.5 0 0;
           0 0 1 0;
           0 0 0 1;
          ];
model.Q = model.Q * Trans_noise_mag;
% Measurement Noise
Meas_noise_mag = 0.5;
%upx^2 = .5, upy^2 = .65, uvx^2 = .4, uvy^2 = .35
model.R = [.5 0;
           0 .65;
          ];
model.R = model.R * Meas_noise_mag;

% Birth parameters
model.t_birth = 7;
model.L_birth = 1;
model.w_birth= zeros(model.L_birth,1);                                % weights of Gaussian birth terms (per model.duration)
model.m_birth= zeros(model.xdim,model.L_birth);                       % means of Gaussian birth terms 
model.P_birth= zeros(model.xdim,model.xdim,model.L_birth);            % cov of Gaussian birth terms
model.p_birth= zeros(3,model.L_birth);                                % shape of Gaussian birth terms
        
model.w_birth = .01;                                         % weight
model.m_birth = [450; 200; -10; -20];                         % kinematic state
model.P_birth = diag([100 100 20 20]);                      % cov
model.p_birth = [56.3/180 * pi; 1 * model.carSize];         % extent state

% Detection parameters
model.P_D = 0.99;       % probability of detection in measurements
model.P_MD= 1-model.P_D; % probability of missed detection in measurements

% Clutter parameters
model.lambda_c = 7; % clutter rate
model.range_c= [0 500; 0 500];      % uniform clutter region
model.pdf_c= 1/prod(model.range_c(:,2)-model.range_c(:,1)); % uniform clutter density

%% Generate groundtruth
model.gt{1} = {[model.gt1(:, 1)]; [model.gt2(:, 1)]};
model.gt_shape{1} = {[model.gt1_shape]; [model.gt2_shape]};

for i = 2:model.duration
    if i >= model.t_birth
        model.gt1(:,i) = model.F * model.gt1(:,i-1) + mvnrnd([0; 0; 0; 0], model.Q, 1)';
        model.gt2(:,i) = model.F * model.gt2(:,i-1) + mvnrnd([0; 0; 0; 0], model.Q, 1)';
        if i == model.t_birth
            model.gt3(:, i) = model.m_birth;
        else
            model.gt3(:, i) = model.F * model.gt3(:, i-1) + mvnrnd([0; 0; 0; 0], model.Q, 1)';
        end

        model.gt{i} = {[model.gt1(:, i)]; [model.gt2(:, i)]; [model.gt3(:, i)]};
        model.gt_shape{i} = {[model.gt1_shape]; [model.gt2_shape]; [model.p_birth]};
    else 
        model.gt1(:,i) = model.F * model.gt1(:,i-1) + mvnrnd([0; 0; 0; 0], model.Q, 1)';
        model.gt2(:,i) = model.F * model.gt2(:,i-1) + mvnrnd([0; 0; 0; 0], model.Q, 1)';

        model.gt{i} = {[model.gt1(:, i)]; [model.gt2(:, i)]};
        model.gt_shape{i} = {[model.gt1_shape]; [model.gt2_shape]};
    end
end

%% Generate measurement
model.z = cell(model.duration, 1);
model.c = cell(model.duration, 1);
model.num_c = zeros(model.duration, 1);

for i = 1:model.duration
    if i >= model.t_birth
        model.num_z{i} = poissrnd(model.lambda_z, [1, 3]);
    else
        model.num_z{i} = poissrnd(model.lambda_z, [1, 2]);
    end

    for k = 1:size(model.num_z{i}, 2)
        for j = 1:model.num_z{i}(k)
            h(j, :) = -1 + 2.* rand(1, 2);
            while norm(h(j, :)) < 1
                h(j, :) = -1 + 2.* rand(1,2);
            end
    
            if rand(1) <= model.P_D
                z_temp = model.gt{i}{k}(1:2) + ...
                h(j, 1) * model.gt_shape{i}{k}(2) * [cos(model.gt_shape{i}{k}(1)); sin(model.gt_shape{i}{k}(1))] + ...
                h(j, 2) * model.gt_shape{i}{k}(3) * [-sin(model.gt_shape{i}{k}(1)); cos(model.gt_shape{i}{k}(1))] + ...
                mvnrnd([0; 0], model.R, 1)';
                model.z{i} = [model.z{i} z_temp];
            end
        end
    end

    model.num_c(i) = poissrnd(model.lambda_c);
    c1 = [unifrnd(model.range_c(1,1),model.range_c(1,2),1,model.num_c(i)); 
        unifrnd(model.range_c(2,1),model.range_c(2,2),1,model.num_c(i))];
    model.c{i} = c1;

    model.z{i} = [model.z{i} model.c{i}];
end

%% Plot groundtruths and measurements
doplot = 0;     % plot ground-truth enable
if doplot
    figure(1);
    hold on;
    gt_plot = plot([model.gt1(1,:), NaN, model.gt2(1,:)], [model.gt1(2,:), NaN, model.gt2(2,:)], '-r.', 'LineWidth', 1.5, 'MarkerSize', 15);
    birth_plot = plot(model.gt3(1, model.t_birth:model.duration), ...
        model.gt3(2, model.t_birth:model.duration), '-r.', 'LineWidth', 1.5, 'MarkerSize', 15);
    for t = 1 : model.duration
        plot_extent([model.gt1(1:2,t); model.gt1_shape], '-', 'r', 1);
        plot_extent([model.gt2(1:2,t); model.gt2_shape], '-', 'g', 1);
        meas_plot = plot(model.z{t}(1,:), model.z{t}(2,:), 'k+', 'MarkerSize', 5);
        clutter_plot = plot(model.c{t}(1, :), model.c{t}(2, :), 'b*', 'MarkerSize', 5);
    end
    xlim([model.range_c(1,1)-10 model.range_c(1,2)+10]);
    ylim([model.range_c(2,1)-10 model.range_c(2,2)+10]);
    xlabel('Position X');
    ylabel('Position Y');
    title('Sensor FOV');
    legend([gt_plot, birth_plot, meas_plot, clutter_plot], 'Ground-truth', 'Birth', 'Measurement', ...
        'Clutter', 'Location', 'southeast');
end