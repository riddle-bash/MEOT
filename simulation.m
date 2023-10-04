% Measurements generating for GM-PHD Multiple Extended Object Tracking
% --------------------------------------------------------------------
% Object numbers: 2
% Transition noise, measurement noise: Gauss
% False Alarm: Poisson
% Birth:

%% Simulation setting
duration = 50;
model = gen_model;

%% Ground-truth, noise setting

Meas_source = cell(duration, 1);
Ground_truth = cell(duration, 1); %Real Possition Of Non-Ego Vehicle
Lin_states = cell(duration, 1);   %Real Possition Of Ego Vehicle

Lin_states{1} = [20; 10; 10; 10];
Ground_truth{1} = [10; 10; 10; 10];

Meas_source{1} = Ground_truth{1} + mvnrnd(zeros(1,model.xdim),model.Q)';

for i = 2:duration
    Lin_states{i} = pagemtimes(model.F, Lin_states{i-1});
    Ground_truth{i} = pagemtimes(model.F, Ground_truth{i-1});
    Meas_source{i} = Ground_truth{i} + mvnrnd(zeros(1,model.xdim),model.Q)';
end

%% Generate measurement
Meas = cell(duration, 1);
O = cell(duration, 1);
C = cell(duration, 1);
obs_noise_mu = [10; 10];
for k = 1:duration
    % Object detections
    if (rand(1) <= model.P_D)
        mu = model.H * Meas_source{k};
        O{k} = mu + mvnrnd(zeros(1,model.zdim),model.R,1)';       % single target observations if detected
    else
        O{k} = [];
    end
    % Clutter
    N_c = poissrnd(model.lambda_c);      % number of clutter points
    model.range_c = model.range_c + Lin_states{k}(1:2);
    model.pdf_c = 1/prod(model.range_c(:,2)-model.range_c(:,1));
    C{k} = repmat(model.range_c(:,1) + Lin_states{k}(1:2),1,N_c) + diag(model.range_c*[-1;1])*rand(model.zdim,N_c); 
    % Measurement is union of detections and clutter
    Meas{k}= [O{k} C{k}];
end

%% Prior
w_update{1}= 1;
m_update{1}= [100;100;10;10];
P_update{1}= diag([100 100 100 100]).^2;
L_update = 1;
est{1} = m_update{1};
num_objects{1} = 1;

% init pruning and merging parameter
elim_threshold = 1e-5;        % pruning threshold
merge_threshold = 4;          % merging threshold
L_max = 100;                  % limit on number of Gaussian components

%% Recursive filtering
for k = 2:duration
    %% Predict
    [m_predict, P_predict] = predict_KF(model, m_update{k-1}, P_update{k-1});
    w_predict = model.P_S * w_update{k-1};
    % Cat with append birth object
    m_predict = cat(2, model.m_birth, m_predict);
    P_predict = cat(3, model.P_birth, P_predict);
    w_predict = cat(1, model.w_birth, w_predict);
    L_predict= model.L_birth + L_update;    %number of objects

    %% Update
    n = size(Meas{k},2);       %number of measurement

    % miss detectection
    w_update{k} = model.P_MD*w_predict;
    m_update{k} = m_predict;
    P_update{k} = P_predict;

    % detection
    [likelihood_tmp] = cal_likelihood(Meas{k},model,m_predict,P_predict);

    if n ~= 0
        [m_temp, P_temp] = update_KF(Meas{k},model,m_predict,P_predict);
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
    w_update{k} = w_update{k}/sum(w_update{k});

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
        est{k}= [ est{k-1} repmat(m_update{k}(:,idx(i)),[1, num_targets]) ];
        num_objects{k} = num_objects{k - 1} + num_targets;
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
figure(1); hold on;
xlabel('x');
ylabel('y');

for k = 2:duration
    lin_plot=plot(Ground_truth{k}(1,:), Ground_truth{k}(2,:), '-rv');
    real_plot=plot(Meas_source{k}(1,:), Meas_source{k}(2,:), 'g^');
    est_plot=plot(est{k}(1,1), est{k}(2,1), 'bo');
    meas_plot=plot(Meas{k}(1,:), Meas{k}(2,:),'k*');
    legend([lin_plot,real_plot,est_plot,meas_plot],{'Groundtruth','RealState','Estimate','Measurement'});
end

%% Evaluation
rms_est_sw = zeros(1,duration);
rms_est_hw = zeros(1,duration);
rms_realstate = zeros(2,duration);
ospa_hw = zeros(1,duration);
for k = 1:duration
   mat_1 = Ground_truth{k}(1:2);

   if (size(est{k},1) ~= 0)
       mat_2 = est{k}(1:2);
   else
       mat_2 = [];
   end
       
   rms_est_hw(:,k) = sqrt(mean((mat_2 - mat_1).^2));
   %ospa_hw(k) = ospa_dist(mat_1,mat_2,100,2);
end

figure(2); hold on; 
tile = tiledlayout('flow');

%plot in tile
nexttile, plot(rms_est_hw, '-bo'); 
title('RMS Estimation');

% nexttile, plot(ospa_hw, '-bo');
% title('OSPA HW Estimation');
% 
% xlabel(tile, 'Time steps', 'FontSize', 16, 'FontWeight', 'bold');
% y_label = ylabel(tile, 'Error', 'FontSize', 16, 'FontWeight', 'bold');
% sgtitle('Error', 'FontWeight', 'bold');


%% 