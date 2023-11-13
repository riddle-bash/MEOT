% -------------Multiple Extended Object Tracking----------------
% ---Using Gaussian Mixture Probability Hypothesis Density----
% ------------------------------------------------------------
% ------------------------------------------------------------
% Scenario 1: 2 objects meet (m = 1)
% Scenario 2: 2 objects not meet (m = 2)
% lambda: poisson rate lambda_c
% gamma: poisson rate to rand measurements of each object num_z

clc, clear, close all;
%% Simulation setting
mode = 2;      % Num of Scenario
model = gen_model(mode);
duration = model.duration;

%% Ground-truth, noise setting

%% Generate measurement
z = model.z;
c = model.c;
lambda_c = model.lambda_c;
lambda_z = model.lambda_z;
pdf_c = model.pdf_c;
num_z = model.num_z;
num_c = model.num_c;

%% Prior
% Kinematic
w_update = cell(1,duration);
m_update = cell(1,duration);
P_update = cell(1,duration);

w_update{1} = [.5 ; .5];
switch mode
    case 1
        m_update{1}(:, 1) = [100; 0; 20; 20];
        P_update{1}(:, :, 1) = diag([100 100 20 30]);
        
        m_update{1}(:, 2) = [100; 500; 15; -20];
        P_update{1}(:, :, 2) = diag([100 100 20 30]);
    case 2
        m_update{1}(:, 1) = [100; 0; 10; 20];
        P_update{1}(:, :, 1) = diag([100 100 20 30]);
        
        m_update{1}(:, 2) = [450; 500; -10; -20];
        P_update{1}(:, :, 2) = diag([100 100 20 30]);
end

est_m = cell(1, duration);
est_P = cell(1, duration);
num_targets = zeros(1, duration);
exec_time = zeros(1, duration);

elim_threshold = 1e-10;
merge_threshold = 4;
L_max = 100;
d_threshold = 600;

% Shape
L_predict = 2;
r{1} = [m_update{1}(:, 1) m_update{1}(:, 2)];
p{1} = repmat([0 model.carSize']', 1, 2);
Cr = repmat(diag([10 10 16 16]), 1, 1, 2);
Cp = repmat(diag([1 .3 .1]), 1, 1, 2);

result_extend = cell(1, duration);
H = [1 0 0 0; 0 1 0 0];
Ar = model.F;
Ap = eye(3);

Ch = diag([1/4 1/4]);
Cv = diag([20 8]);
Cwr = diag([10 10 1 1]);
Cwp = diag([.05 .001 .001]);

for k = 1:duration
    execution = tic;
%% Kinematic predict

    if k == 1
        w_predict = w_update{k};
        m_predict = m_update{k};
        P_predict = P_update{k};
    else
        [m_predict, P_predict] = predict_KF(model, m_update{k-1}, P_update{k-1});
        w_predict = model.P_S * w_update{k-1};
    end

    w_predict = cat(1, model.w_birth, w_predict);
    m_predict = cat(2, model.m_birth, m_predict);
    P_predict = cat(3, model.P_birth, P_predict);
    L_predict = L_predict + model.L_birth;

%% Partitioning
    P = {};
    num_meas = size(z{k}, 2);
    Meas = z{k};
    for i = 20:100:d_threshold+1
        % Distance partitioning for each Partition P
        P{end + 1} = distance_partitioning(i, Meas);
    end
%     P = removeDuplicatePartitions(P);

%% Kinematic update
    %%% Miss detection
    w_update{k} = (1 - (1 - exp(-lambda_z))*model.P_D) * w_predict;
    m_update{k} = m_predict;
    P_update{k} = P_predict;

    %%% Detection
    % Calculate parameters for update
    dW_temp = {};   % init for cell weight computation
    dW = {};        % weight of cell
    l = {};         % likelihood of measurement
    l_cell = {};         % likelihood of cell

    % Set of Partiton P
    for n = 1 : size(P, 2)
        % Set of Cell W in Partition P{n}
        for m = 1 : size(P{n}.W, 2)
            W_meas = P{n}.W{m}.Meas;     % Set of measurements in Cell W{m}

            % Likelihood of all measurements in W_meas with m_predict and P_predict
            l{n}{m} = cal_likelihood(W_meas, model, m_predict, P_predict);
            l_tmp = l{n}{m} / (lambda_c * pdf_c);  % likelihood (7)
            l_tmp = prod(l_tmp, 2);

            gamma = exp(-lambda_z) * lambda_z^size(W_meas, 2);     % gamma (12c)
            dW_temp{n}{m} = gamma * model.P_D * w_predict .* l_tmp;
            dW{n}(m) = (size(W_meas, 2) == 1) + sum(dW_temp{n}{m}, 'all'); % (12j)

            P{n}.W{m}.likelihood = l{n}{m};
            P{n}.W{m}.l_tmp = l_tmp;
            P{n}.W{m}.gamma = gamma;
            P{n}.W{m}.dW_temp = dW_temp{n}{m};
            P{n}.W{m}.dW = dW{n}(m);
            P{n}.W{m}.m_predict = m_predict;
        end
    end

    sum_tmp = 0;
    for n_P = 1 : size(P, 2)
        sum_tmp = sum_tmp + sum(dW{n_P}, 'all');   % sum of all partition weight
    end

    % Normalize partition weight
    w_P = zeros(1, size(P, 2));
    for n_P = 1 : size(P, 2)
        w_P(n_P) = sum(dW{n_P}, 'all') / sum_tmp;  % weight of each partition (12i)
        P{n_P}.w_P = w_P(n_P);
    end

    % Update kinematic
    for n_P = 1 : size(P, 2)
        P{n_P}.w_P = w_P(n_P);
        % W_card = size(P{n_P}.W, 2);
        for m_W = 1 : size(P{n_P}.W, 2)
            W_meas = P{n_P}.W{m_W}.Meas;     % Set of measurements in Cell W{m}
            if ~isempty(W_meas)
                [m_temp, P_temp] = update_KF(W_meas, model, m_predict, P_predict);
                l_tmp = l{n_P}{m_W} / (lambda_c * pdf_c);  % likelihood (12d)
                l_tmp = prod(l_tmp, 2);

                gamma = exp(-lambda_z) * lambda_z^size(W_meas, 2);       % gamma (12c)
                w_temp = w_P(n_P) * gamma * model.P_D * w_predict .* l_tmp / dW{n_P}(m_W); % weight (12b)
                
                P{n_P}.W{m_W}.w_temp = w_temp;
                for idx = 1 : size(w_temp, 1)
                   w_update{k} = cat(1, w_update{k}, w_temp(idx));
                   m_update{k} = cat(2, m_update{k}, m_temp(:, idx));
                   P_update{k} = cat(3, P_update{k}, P_temp);
                end
            end
        end
    end
    
    [w_update{k}, m_update{k}, P_update{k}] = gaus_prune(w_update{k}, m_update{k}, P_update{k}, elim_threshold);
    [w_update{k}, m_update{k}, P_update{k}] = gaus_merge(w_update{k}, m_update{k}, P_update{k}, merge_threshold);
    [w_update{k}, m_update{k}, P_update{k}] = gaus_cap(w_update{k}, m_update{k}, P_update{k}, L_max);

    num_targets(k) = floor(sum(w_update{k}));
    w_copy = w_update{k};

    indices = [];

    for i = 1:num_targets(k)
        [~, maxIndex] = max(w_copy);
        indices(i) = maxIndex;
        w_copy(maxIndex) = -inf;
    end
    
    for i = 1:size(indices, 2)
        est_m{k} = [est_m{k} m_update{k}(:, indices(i))];
        est_P{k} = cat(3, est_P{k}, P_update{k}(:, :, indices(i)));
    end

    result{k}.Partition = P;
    result{k}.w_P = w_P;
    result{k}.est = est_m{k};

%     figure;
%     hold on;
%     gt_plot1 = plot(model.gt1(1, k), model.gt1(2, k), 'r^', 'MarkerSize', 10);
%     gt_plot2 = plot(model.gt2(1, k), model.gt2(2, k), 'ro', 'MarkerSize', 10);
%     for i = 1:size(m_predict, 2)
%         predict_plot = plot(m_predict(1, i), m_predict(2, i), 'b+');
%     end
%     for i = 1:num_targets(k)
%         est_plot = plot(est_m{k}(1, i), est_m{k}(2, i), 'b*');
%     end
%     [~, P_idx] = max(w_P);
%     [~, W_idx] = min(dW{P_idx});
%     disp(dW{P_idx}(W_idx));
% 
%     cell_tmp = P{P_idx}.W{W_idx}.Meas;
%     disp(cell_tmp);
%     for i = 1:size(cell_tmp, 2)
%         cell_plot = plot(cell_tmp(1, i), cell_tmp(2, i), 'kv');
%     end
% 
%     title('Partitioning and Estimate');
%     axis equal
%     legend([gt_plot1, gt_plot2, predict_plot, est_plot, cell_plot], 'Ground-truth 1', 'Ground-truth 2', ...
%         'Predict', 'Estimation', 'Cell', 'Location', 'southeast');
%     close all;

%% Shape update & Shape predict
    % Get suitable measurement set for estimated state
%     for i = 1 : size(w_P, 2)
%         l_partition(i) = P{i}.w_P * sum(P{i}.l_cell, 'all');
%     end
%     
%     [~, P_idx] = max(l_partition);
%     [l_cell_val, l_cell_idx] = sort(P{P_idx}.l_cell, 'descend');

    [~, P_idx] = max(w_P);
    ET_meas{k} = P{P_idx}.W;

    r{k} = zeros(4, num_targets(k));
    for i = 1:num_targets(k)
        r{k}(:, i) = est_m{k}(:, i);
        Cr(:, :, i) = est_P{k}(:, :, i);
        if i > size(p{k}, 2)
            p{k}(:, i) = [0 model.carSize']';
            Cp(:, :, i) = diag([1 .3 .1]);
        end
    end

    % Sort Meas, r
    [ET_meas{k}, r{k}, p{k}] = assignment(ET_meas{k}, r{k}, p{k}, num_targets(k));
    
    for i = 1:num_targets(k)
        Meas = ET_meas{k}{i};

%         [r{k}(:, i), p{k}(:, i), Cr(:, :, i), Cp(:, :, i)] = measurement_update(Meas, H, ...
%             r{k}(:, i), p{k}(:, i), Cr(:, :, i), Cp(:, :, i), Ch, Cv);
% 
%         [r{k+1}(:, i), p{k+1}(:, i), Cr(:, :, i), Cp(:, :, i)] = time_update(r{k}(:, i), p{k}(:, i), ...
%             Cr(:, :, i), Cp(:, :, i), Ar, Ap, Cwr, Cwp);
        [p{k}(:, i), Cp(:, :, i)] = shape_update(Meas, H, r{k}(:, i), p{k}(:, i), ...
            Cr(:, :, i), Cp(:, :, i), Ch, Cv);

        [p{k+1}(:, i), Cp(:, :, i)] = shape_predict(p{k}(:, i), Cp(:, :, i), Ap, Cwp);
    end

    result_extend{k}.r = r{k};
    result_extend{k}.p = p{k};
    result_extend{k}.meas = ET_meas{k};
    result_extend{k}.est_m = est_m{k};
    exec_time(k) = toc(execution);
    disp(['Time step ', num2str(k), ...
        ': measurements = ', num2str(size(z{k}, 2)), ...
        ', Partitions = ', num2str(size(P, 2)), ...
        ', estimations = ', num2str(num_targets(k)), ...
        ', reflection points = ', num2str(num_z{k}), ...
        ', clutters = ', num2str(num_c(k)), ...
        ', execution time = ', num2str(exec_time(k)) , 's']);
end

%% Plot and visualize
figure;
hold on
gt_plot = plot([model.gt1(1,:), NaN, model.gt2(1,:)], [model.gt1(2,:), NaN, model.gt2(2,:)], '-r.', 'LineWidth', 1.5, 'MarkerSize', 15);
birth_plot = plot(model.gt3(1, model.t_birth:model.duration), ...
        model.gt3(2, model.t_birth:model.duration), '-r.', 'LineWidth', 1.5, 'MarkerSize', 15);
for t = 1:model.duration
    
    for k = 1:num_targets(t)
        est_plot = plot(est_m{t}(1, k), est_m{t}(2, k), 'b*');
    end

    meas_plot = plot(model.z{t}(1,:), model.z{t}(2,:), 'k.', 'MarkerSize', 1);
end

xlim([model.range_c(1,1) model.range_c(1,2)]+10);
ylim([model.range_c(2,1) model.range_c(2,2)+10]);
xlabel('Position X');
ylabel('Position Y');
title('GM-PHD Estimation');
legend([gt_plot, birth_plot, est_plot, meas_plot], 'Ground-truth', 'Birth', 'Estimation', 'Measurement', 'Location', 'bestoutside');

figure (3);
hold on;

for t = 1:duration
    if mod(t, 1) == 0
        gt_center_plot1 = plot(model.gt1(1,t), model.gt1(2,t), 'r.');
        gt_center_plot2 = plot(model.gt2(1,t), model.gt2(2,t), 'r.');
        gt_plot1 = plot_extent([model.gt1(1:2,t); model.gt1_shape], '-', 'r', 1);
        gt_plot2 = plot_extent([model.gt2(1:2,t); model.gt2_shape], '-', 'g', 1);
        if t >= model.t_birth
            gt_center_plot3 = plot(model.gt3(1,t), model.gt3(2,t), 'r.');
            gt_plot3 = plot_extent([model.gt3(1:2,t); model.p_birth], '-', 'y', 1);
        end

        for n = 1 : num_targets(t)
           est_center_plot = plot(r{t}(1, n), r{t}(2, n), 'b+');
           est_plot = plot_extent([r{t}(1:2, n); p{t}(:, n)], '-', 'b', 1);
           meas_plot = plot(result_extend{t}.meas{n}(1, :), result_extend{t}.meas{n}(2, :), 'k^');
        end
    end
end

xlim([model.range_c(1,1) 600]);
ylim([model.range_c(2,1) 600]);
xlabel('Position X');
ylabel('Position Y');
title('Extended GM-PHD Estimation');
legend([gt_plot1, gt_plot2, gt_plot3, est_plot], 'Ground-truth 1', 'Ground-truth 2', 'Birth', 'Estimation', 'Location', 'southeast');

%% Evaluation
doPlotOSPA_kinematic = false;

if doPlotOSPA_kinematic
    disp(['------------------------------Total run time: ', num2str(sum(exec_time, 2)), 's---------------------------']);
    ospa = zeros(1, duration);
    ospa_cutoff = 200;
    ospa_order = 2;
    
    for t = 1:duration
        if t >= model.t_birth
            gt{t} = [model.gt1(:, t), model.gt2(:, t), gt3(:, t)];
        else
            gt{t} = [model.gt1(:, t), model.gt2(:, t)];
        end
        gt_mat = gt{t};
        %[gt_mat, est_mat] = get_uniform_points_boundary(gt{t}{i}, [r{t}(1:2,i); p{t}(:,i)]', 50);
        est_mat = est_m{t};
        ospa(t) = ospa_dist(gt_mat, est_mat, ospa_cutoff, ospa_order);
    end

    figure (4);
    hold on;
    
    plot(2:duration, ospa(2:end));

    xlabel('Time step');
    ylabel('Distance (in m)');
    title('OSPA Evaluation');
end

doPlotOSPA_extend = true;
disp(['------------------------------Total run time: ', num2str(sum(exec_time, 2)), 's---------------------------']);

if doPlotOSPA_extend
    ospa = zeros(1, duration);
    ospa_cutoff = 200;
    ospa_order = 3;
    
    gt_mat = cell(1, duration);
    est_mat = cell(1, duration);
    for t = 1:duration
        if t >= model.t_birth
            gt{t} = {[model.gt1(1:2, t); model.gt1_shape]'; [model.gt2(1:2, t); model.gt2_shape]'; ...
                [model.gt3(1:2, t); model.p_birth]'};
        else
            gt{t} = {[model.gt1(1:2, t); model.gt1_shape]'; [model.gt2(1:2, t); model.gt2_shape]'};
        end
        for i = 1:num_targets(k)
            if i > size(gt{t}, 1)
                gt{t}{end + 1} = [];
            end
            if i > size(r{t}, 2)
                est{i} = [];
            else
                est{i} = [r{t}(1:2,i); p{t}(:,i)]';
            end
            [gt_mat_tmp, est_mat_tmp] = get_uniform_points_boundary(gt{t}{i}, est{i}, 50);
            gt_mat{t} = [gt_mat{t}, gt_mat_tmp];
            est_mat{t} = [est_mat{t}, est_mat_tmp];
        end
        ospa(t) = ospa_dist(gt_mat{t}, est_mat{t}, ospa_cutoff, ospa_order);
    end

    figure (4);
    hold on;
    plot(1:duration, ospa(1:end));

    ylim([0 ospa_cutoff]);
    xlim([1 duration]);
    xlabel('Time step');
    ylabel('Distance (in m)');
    title('OSPA Evaluation');
end