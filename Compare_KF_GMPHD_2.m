% Comparison Kalman Filter & GM-PHD Filter
% --------------------------------------------------------------------
% Object numbers: 1
% Transition noise, measurement noise: Gauss
% False Alarm: Poisson
% No Death & Birth Model Involved
% Object start from [0,0] to [495,620] in 40 time steps (40 seconds)

clc, clear, close all;
%% Simulation setting
duration = 40;
model = gen_model;

%% Ground-truth, noise setting
 
gt(:,1) = [0;0;12;15];

for i = 2:duration
    gt(:,i) = model.F * gt(:,i-1);
end

%% Generate measurement
model.lambda_c = 10; % clutter rate
model.range_c= [-10 490; -10 620];      % uniform clutter region
model.pdf_c= 1/prod(model.range_c(:,2)-model.range_c(:,1)); % uniform clutter density

for i = 1:duration
    %Gen Observation
    z{i} = repmat(model.H * gt(:, i),1,1) + mvnrnd(0,1,2,1);

    %Gen Clutter
    num_clutters = 10;
    c(:,:,i) = [unifrnd(-10,490,1,num_clutters);unifrnd(-10,620,1,num_clutters)];

    z{i} = [z{i} c(:, :, i)];
end

%% Prior
% Kalman
KM_m_update{1}(:, 1) = [10; 10; 10; 10];
KM_P_update{1}(:, :, 1) = diag([100 100 100 100]).^2;

% GM-PHD
w_update{1} = 0.5;
m_update{1}(:, 1) = [100; 100; 10; 10];
P_update{1}(:, :, 1) = diag([100 100 100 100]).^2;
% D{1} = gmdistribution(m_update{1}, P_update{1}, w_update{1});
L_update = 1;
est = cell(duration, 1);
num_objects = zeros(duration, 1);

% init pruning and merging parameter
elim_threshold = 1e-5;        % pruning threshold
merge_threshold = 4;          % merging threshold
L_max = 100;                  % limit on number of Gaussian components

%% Recursive filtering
for k = 2:duration
    %% Predict
    % Kalman
    [KM_m_predict, KM_P_predict] = predict_KF(model, KM_m_update{k-1}, KM_P_update{k-1});
    % GM-PHD
    [m_predict, P_predict] = predict_KF(model, m_update{k-1}, P_update{k-1});
    w_predict = model.P_S * w_update{k-1};
    % Cat with append birth object
    m_predict = cat(2, model.m_birth_3, m_predict);
    P_predict = cat(3, model.P_birth_3, P_predict);
    w_predict = cat(1, model.w_birth_3, w_predict);
%     L_predict= model.L_birth + L_update;    %number of objects

    %% Update
    n = size(z{k},2);       %number of measurement
    % Kalman
    for i = 1:n
        [KM_m_update{k}, KM_P_update{k}] = update_single(z{k}, model.H, model.R, KM_m_predict, KM_P_predict);
    end
    
    % GM-PHD

    % miss detectection
    w_update{k} = model.P_MD*w_predict;
    m_update{k} = m_predict;
    P_update{k} = P_predict;

    % detection
    [likelihood_tmp] = cal_likelihood(z{k},model,m_predict,P_predict);

    if n ~= 0
        [m_temp, P_temp] = update_KF(z{k},model,m_predict,P_predict);
        for i = 1:n
            % Calculate detection weight of each probable object detect
            w_temp = model.P_D * w_predict .* likelihood_tmp(:,i);
            w_temp = w_temp ./ (model.lambda_c*model.pdf_c + sum(w_temp));
            % Cat all of them to a vector of weight
            w_update{k} = cat(1,w_update{k},w_temp);
            % Update mean and covariance
            m_update{k} = cat(2,m_update{k},m_temp(:,:,i));
            P_update{k} = cat(3,P_update{k},P_temp);
        end
    end

    %normalize weights
    %w_update{k} = w_update{k}/sum(w_update{k});

    %---mixture management
    L_posterior= length(w_update{k});
    
    % pruning, merging, caping
    [w_update{k},m_update{k},P_update{k}]= gaus_prune(w_update{k},m_update{k},P_update{k},elim_threshold);    
    L_prune= length(w_update{k});
    [w_update{k},m_update{k},P_update{k}]= gaus_merge(w_update{k},m_update{k},P_update{k},merge_threshold);   
    L_merge= length(w_update{k});
    [w_update{k},m_update{k},P_update{k}]= gaus_cap(w_update{k},m_update{k},P_update{k},L_max);               
    L_cap= length(w_update{k});
    
    L_update= L_cap;

    % Estimate x
    num_objects(k) = round(sum(w_update{k}));
    num_targets = num_objects(k);
    w_copy = w_update{k};
    indices = [];

    for i = 1:num_objects(k)
        [~, maxIndex] = max(w_copy);
        indices(i) = maxIndex;
        w_copy(maxIndex) = -inf;
    end

    for i = 1:size(indices,2)
        est{k} = [est{k} m_update{k}(:,i)];
    end

    %---display diagnostics
    disp([' time= ',num2str(k),...
         ' #gaus orig=',num2str(L_posterior),...
         ' #gaus elim=',num2str(L_prune), ...
         ' #gaus merg=',num2str(L_merge), ...
         ' #gaus cap=',num2str(L_cap), ...
         ' #measurement number=',num2str(n)]);
end


%% Plot and visualize
figure(1); 
hold on;
gt_plot = plot(gt(1,:),gt(2,:),'-r.','LineWidth',1.5,'MarkerSize',15);
for t = 2:duration
    for k = 1:num_objects(t)
        est_plot = plot(est{t}(1, k), est{t}(2, k), 'b*','MarkerSize',10);
    end
    meas_plot = plot(z{t}(1, :), z{t}(2, :), 'k.','MarkerSize', 5);
    KM_plot = plot(KM_m_update{t}(1, 1), KM_m_update{t}(2, 1), 'go');
end
xlim([-10 500]);
ylim([-1 640]);
xlabel('Position X');
ylabel('Position Y');
title('Tracking Estimation');
legend([est_plot, KM_plot, gt_plot, meas_plot],'GM-PHD','Kalman Filter','Ground-truth', 'Measurement', 'Location','southeast');

figure(2); 
hold on;
for t = 1:duration
    meas_plot = plot(z{t}(1, :), z{t}(2, :), 'k.','MarkerSize',5);
end
gt_plot = plot(gt(1,:),gt(2,:),'-r.','LineWidth',1.5,'MarkerSize',15);
xlim([-10 500]);
ylim([-1 640]);
xlabel('Position X');
ylabel('Position Y');
title('Sensor FOV');
legend([gt_plot, meas_plot],'Ground-truth', 'Measurement', 'Location','southeast');

%% Evaluation
figure(3);
hold on;
for t = 1:duration
    if ~isempty(est{t})
        ospa1(t) = ospa_dist(KM_m_update{t}(1:2,:),gt(1:2,t),30,1);
        ospa2(t) = ospa_dist(est{t}(1:2,1),gt(1:2,t),30,1);
    else
        ospa1(t) = ospa_dist([0; 0],gt(1:2,t),30,1);
        ospa2(t) = ospa_dist([0; 0],gt(1:2,t),30,1);
    end
end
hold on;
ospa_KF = plot(1:duration, ospa1 , 'g','LineWidth',1.5);
ospa_PHD = plot(1:duration, ospa2, 'b','LineWidth',1.5);
xlim([0 40]);
ylim([-1 37]);
xlabel('Time step');
ylabel('Distance');
title('OSPA Evaluation')
legend([ospa_KF, ospa_PHD], 'Kalman Filter', 'GM-PHD Filter', 'Location', 'northeast');

%% 