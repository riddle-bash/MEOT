% -------------Multiple Extended Object Tracking----------------
% ---Using Gaussian Mixture Probability Hypothesis Density----
% ------------------------------------------------------------
% ------------------------------------------------------------
% Scenario 1: 2 objects not meet (m = 1)
% Scenario 2: 2 objects meet (m = 2)
% lambda: poisson rate lambda_c
% gamma: poisson rate to rand measurements of each object num_z

clc, clear, close all;

%% Multi-simulation
multi_num_loop = 100;
mean_kinematic_ospa = zeros(1, 20);
mean_extend_ospa = zeros(1, 20);
for i_loop = 1:multi_num_loop
    disp(['------------------------------Loop steps: ', num2str(i_loop) , '---------------------------']);
    %% Simulation setting
    mode = 2;      % Num of Scenario
    model = gen_model(mode);
    duration = model.duration;
    
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
            m_update{1}(:, 1) = [100; 100; 10; 10];
            P_update{1}(:, :, 1) = diag([100 100 20 30]);
    
            m_update{1}(:, 2) = [100; 200; 10; 10];
            P_update{1}(:, :, 2) = diag([100 100 20 30]);
        case 2
            m_update{1}(:, 1) = [200; 100; -10; 10];
            P_update{1}(:, :, 1) = diag([100 100 20 30]);
    
            m_update{1}(:, 2) = [200; 200; 10; 10];
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
    r = cell(1, duration);
    p = cell(1, duration);
    r{1} = [m_update{1}(:, 1) m_update{1}(:, 2)];
    p{1} = repmat([0 1.5*model.carSize']', 1, 2);
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

    % OSPA evaluation
    kinematic{i_loop}.ospa = zeros(1, duration);
    kinematic{i_loop}.ospa_cutoff = 200;
    kinematic{i_loop}.ospa_order = 2;
    kinematic{i_loop}.gt_mat = cell(1, duration);
    kinematic{i_loop}.est_mat = cell(1, duration);

    extend{i_loop}.ospa = zeros(1, duration);
    extend{i_loop}.ospa_cutoff = 200;
    extend{i_loop}.ospa_order = 3;
    extend{i_loop}.gt_mat = cell(1, duration);
    extend{i_loop}.est_mat = cell(1, duration);

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
    
        num_targets(k) = round(sum(w_update{k}));
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
    
    %% Shape update & Shape predict
        % Get suitable measurement set for estimated state
    
        [~, P_idx] = max(w_P);
        ET_meas{k} = P{P_idx}.W;
    
        r{k} = zeros(4, num_targets(k));
        for i = 1:num_targets(k)
            r{k}(:, i) = est_m{k}(:, i);
            Cr(:, :, i) = est_P{k}(:, :, i);
            if i > size(p{k}, 2)
                p{k}(:, i) = [0 1.5*model.carSize']';
                Cp(:, :, i) = diag([1 .3 .1]);
            end
        end
    
        % Sort Meas, r
        [ET_meas{k}, r{k}, p{k}] = assignment(ET_meas{k}, r{k}, p{k}, num_targets(k));
        est_extend{k} = {};
        gt_extend{k} = {};
        
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
            est_extend{k}{i} = [r{k}(1:2, i); p{k}(:, i)];
        end
    
        if k >= model.t_birth
            gt_extend{k} = {[model.gt1(1:2, k); model.gt1_shape] [model.gt2(1:2, k); model.gt2_shape] ...
                [model.gt3(1:2, k); model.p_birth]};
        else
            gt_extend{k} = {[model.gt1(1:2, k); model.gt1_shape] [model.gt2(1:2, k); model.gt2_shape]};
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
        %% Evaluation
        doPlotOSPA_kinematic = true;
        
        if doPlotOSPA_kinematic
            
            if k >= model.t_birth
                kinematic{i_loop}.gt{k} = [model.gt1(:, k), model.gt2(:, k), model.gt3(:, k)];
            else
                kinematic{i_loop}.gt{k} = [model.gt1(:, k), model.gt2(:, k)];
            end
            kinematic{i_loop}.gt_mat{k} = kinematic{i_loop}.gt{k};
            kinematic{i_loop}.est_mat{k} = est_m{k};
            kinematic{i_loop}.ospa(k) = ospa_dist(kinematic{i_loop}.gt_mat{k}, kinematic{i_loop}.est_mat{k}, ...
                kinematic{i_loop}.ospa_cutoff, kinematic{i_loop}.ospa_order);
        end
        
        doPlotOSPA_extend = true;
        
        if doPlotOSPA_extend
            extend{i_loop}.ospa(k) = 0;
            % Optimal assignment between estimate and groundtruth
            if num_targets(k) ~= 0
                [est_extend{k}, gt_extend{k}, dim] = est_assignment(est_extend{k}, gt_extend{k});
                for i = 1:dim
                    if isempty(gt_extend{k}{i}) || isempty(est_extend{k}{i})
                        extend{i_loop}.ospa(k) = extend{i_loop}.ospa(k) + extend{i_loop}.ospa_cutoff;
                    else
                        [gt_mat_tmp, est_mat_tmp] = get_uniform_points_boundary(gt_extend{k}{i}', est_extend{k}{i}', 50);
                        extend{i_loop}.ospa(k) = extend{i_loop}.ospa(k) + ...
                            ospa_dist(gt_mat_tmp, est_mat_tmp, extend{i_loop}.ospa_cutoff, extend{i_loop}.ospa_order);
                    end
                end
                extend{i_loop}.ospa(k) = extend{i_loop}.ospa(k) / dim;
            else
                extend{i_loop}.ospa(k) = extend{i_loop}.ospa_cutoff;
            end
        end
    end
    disp(['------------------------------Total run time: ', num2str(sum(exec_time, 2)), 's---------------------------']);
    mean_kinematic_ospa = mean_kinematic_ospa + kinematic{i_loop}.ospa;
    mean_extend_ospa = mean_extend_ospa + extend{i_loop}.ospa;
end

%% Plot and visualize
mean_kinematic_ospa = mean_kinematic_ospa / multi_num_loop;
mean_extend_ospa = mean_extend_ospa / multi_num_loop;
figure(1);
hold on;
ospa_kin_plot = plot(1:duration, mean_kinematic_ospa(1:end));

ylim([0 200]);
xlim([1 duration]);
xlabel('Time step');
ylabel('Distance (in m)');
title('OSPA Evaluation of Kinematic');

figure (2);
hold on;
ospa_ext_plot = plot(1:duration, mean_extend_ospa(1:end));

ylim([0 200]);
xlim([1 duration]);
xlabel('Time step');
ylabel('Distance (in m)');
% title('OSPA Evaluation of Extended Target');
legend(ospa_ext_plot,'OSPA distance','Location','best');