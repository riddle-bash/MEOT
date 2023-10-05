% Measurements generating for GM-PHD Multiple Extended Object Tracking
% --------------------------------------------------------------------
% Object numbers: 2
% Transition noise, measurement noise: Gauss
% False Alarm: Poisson
% Birth:

clc, clear;
%% Simulation setting
duration = 100;
model = gen_model;

%% Ground-truth, noise setting

gt1(:,1) = [-250;-250;12;-2.5];
gt2(:,1) = [250;250;2.5;-12];

for i = 2:duration
    gt1(:,i) = model.F * gt1(:,i-1);
    gt2(:,i) = model.F * gt2(:,i-1);
end

%% Generate measurement
for i = 1:duration
    z1 = gt1(1:2,i);
    z2 = gt2(1:2,i);

    region = [-1000,1000;
        -1000, 1000];
    c(:,:,i) = [unifrnd(-1000,1000,1,50);unifrnd(-1000,1000,1,50)];

    z{i} = [z1 z2 c(:, :, i)];
end

%% Prior
w_update{1} = [0.5; 0.5];
m_update{1}(:, 1) = [100; 100; 10; 10];
P_update{1}(:, :, 1) = diag([100 100 100 100]).^2;
m_update{1}(:, 2) = [200; 100; 10; 10];
P_update{1}(:, :, 2) = diag([100 100 100 100]).^2;
% D{1} = gmdistribution(m_update{1}, P_update{1}, w_update{1});
L_update = 2;
est = cell(duration, 1);
num_objects = zeros(duration, 1);

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
    n = size(z{k},2);       %number of measurement

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
subplot(311);
hold on;
plot(gt1(1,:),gt1(2,:));
plot(gt2(1,:),gt2(2,:));
xlim([-400 1000]);
ylim([-1000 400]);

subplot(312);
hold on;
plot(1:duration,gt1(1,:),'k.');
plot(1:duration,gt2(1,:),'k.');
for t = 1:duration
plot(t,c(1,:,t),'kx');
end

subplot(313);
hold on;
plot(1:duration,gt1(2,:),'k.');
plot(1:duration,gt2(2,:),'k.');
for t = 1:duration
plot(t,c(2,:,t),'kx');
end


figure(2); hold on;
for t = 2:duration
    for k = 1:num_objects(t)
        plot(est{t}(1, k), est{t}(2, k), 'b*');
    end
end
plot(gt1(1,:),gt1(2,:));
plot(gt2(1,:),gt2(2,:));
%% Evaluation


%% 