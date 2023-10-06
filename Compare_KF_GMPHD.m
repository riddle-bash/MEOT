% Measurements generating for GM-PHD Multiple Extended Object Tracking
% --------------------------------------------------------------------
% Object numbers: 2
% Transition noise, measurement noise: Gauss
% False Alarm: Poisson
% Birth:

clc, clear, close all;
%% Simulation setting
duration = 10;
model = gen_model;

%% Ground-truth, noise setting
 
gt(:,1) = [-250;-250;2;3];

for i = 2:duration
    gt(:,i) = model.F * gt(:,i-1);
end

%% Generate measurement
for i = 1:duration
    z = repmat(model.H * gt(:, i),1,1) + mvnrnd(zeros(1,model.zdim),model.R,1)';
    region = [-1000,1000;
        -1000, 1000];
    c(:,:,i) = [unifrnd(-1000,1000,1,50);unifrnd(-1000,1000,1,50)];
    z{i} = z;
    z{i} = cat(2, z{i}, c(:, :, i));
end

%% Prior
% Kalman
KM_m_update{1}(:, 1) = [100; 100; 10; 10];
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
%     m_predict = cat(2, model.m_birth, m_predict);
%     P_predict = cat(3, model.P_birth, P_predict);
%     w_predict = cat(1, model.w_birth, w_predict);
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
            w_temp = w_temp ./ (sum(w_temp));
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
    idx = find(w_update{k} > 0.5 );
    for i = 1:length(idx)
        %num of targets in each density
        num_targets = round(w_update{k}(idx(i)));
        est{k}= [ est{k} repmat(m_update{k}(:,idx(i)),[1, num_targets]) ];
        num_objects(k) = num_objects(k) + num_targets;
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
subplot(211);
hold on;
for t = 1:duration
    if ~isempty(est{t})
        plot(t,est{t}(1,:),'kx');
    end
    plot(t,gt(1,t),'b.');
end
ylabel('X coordinate (in m)');
xlabel('time step');

subplot(212);
hold on;
for t = 1:duration
    if ~isempty(est{t})
        plot(t,est{t}(2,:),'kx');
    end
    plot(t,gt(2,t),'b.');
end
ylabel('Y coordinate (in m)');
xlabel('time step');


figure(2); 
hold on;
for t = 2:duration
    for k = 1:num_objects(t)
        est_plot = plot(est{t}(1, k), est{t}(2, k), 'b*');
    end
    meas_plot = plot(z{t}(1, :), z{t}(2, :), 'k.');
    KM_plot = plot(KM_m_update{t}(1, 1), KM_m_update{t}(2, 1), 'kh');
end
gt_plot = plot(gt(1,:),gt(2,:));

legend([est_plot, gt_plot, meas_plot],'Estimations','Ground-truth', 'Measurement','Location','northeast');
%legend([est_plot],'Estimations','Location','northeast');

%% Evaluation
% figure(3);
% hold on;
% for t = 1:duration
%     if ~isempty(est{t})
%         ospa1(t) = ospa_dist(est{t}(1:2,1),gt1(1:2,t),200,1);
%     else
%         ospa1(t) = ospa_dist([0; 0],gt1(1:2,t),200,1);
%     end
% end
% 
% plot(1:duration, ospa1);

%% 