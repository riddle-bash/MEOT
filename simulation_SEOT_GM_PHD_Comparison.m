% ----------Comparison for Multiple-runtimes and Long-duration-------
% ----------------of Single Extended Object Tracking-----------------
% -------------------Using GM-PHD and MEM-EKF------------------------
% ===================================================================
% |  |     Simulation includes      |          Distribution         |
% |==|===============================================================
% |1.|  Random reflections points   |           Poissons            |
% |2.|   Random miss-detection      |           Uniform             |
% |3.|      Random clutters         |           Poissons            |
% |4.|     Observation Noises       |           Gaussian            |
% ===================================================================
% |    Scenario 1      |      Scenario 2      |      Scenario 3     |
% ===================================================================
% |             o      |         o            |       o  o  o       |
% |           /        |       /   \          |    o           o    |
% |         /          |     /       \        |  o               o  |
% |       /            |   o           \      |  o               o  |
% |     /              |                 \    |    o           o    |
% |   o                |                   o  |       o  o  o       |
% ===================================================================
% |           Important NOTE for 'Setting' the parameters           |
% ===================================================================
% |           To improve estimation:        |    To be realistic:   |
% |1. Increase duration                     |                       |
% |2. Increase dectection probability       |  Inverse the Options  |
% |3. Decrease noise amplifier              |     on the Left       |
% |4. Decrease partition distance threshold |                       |
% |                                         |                       |
% ===================================================================

clc, clear, close all
%% Multi-simulation setting
multi_num_loop = 10;
multi_simu_dur = 100;

custom_scenario = 1;

doPlotOSPA = true;
doPlotAverageOSPA = true;
doPlot_SensorFOV = false; 
doPlot_Estimation = false;
doPlot_ExtendedEstimation = false;

doPlotExtended_SensorFOV = true;
doPlotExtended_Estimation = true;

model = gen_model;
duration = multi_simu_dur;
scenarioID = custom_scenario;

%% Ground-truth and noise setting
[gt_dynamic, gt_orient] = ground_truth_generate(scenarioID, duration, model.F, false);
gt_shape = [gt_orient; repmat(model.carSize, 1, duration)];

%% Measurements setting
z = cell(duration, 1);
c = cell(duration, 1);
num_z = zeros(duration, 1);
num_c = zeros(duration, 1);
lambda_z = 5;
lambda_c = 50;
noise_amp = 1;
detection_probability = 1;

clutter_region = [gt_dynamic(1, 1) - 10  gt_dynamic(1, end) + 10;
                    gt_dynamic(2, 1) - 10 gt_dynamic(2, end) + 10];
pdf_c = 1/prod(clutter_region(:, 2) - clutter_region(:, 1));
                
%% Tolerance setting
elim_threshold = 1e-5;
merge_threshold = 4;
L_max = 100;
d_threshold = 20;

%% Evaluation setting
multi_ospa = zeros(multi_num_loop, multi_simu_dur);
multi_exec_time = zeros(multi_num_loop, multi_simu_dur);

disp(['---------------------Multi-simulation Runtime--------------------']);

%% Multi-simulation
for i_loop = 1:multi_num_loop
    single_exec_time = zeros(duration, 1);

%% Prior
    w_update = cell(duration, 1);
    m_update = cell(duration, 1);
    P_update = cell(duration, 1);
    
    est = cell(duration, 1);
    num_targets = zeros(duration, 1);

    w_update{1} = .01;
    m_update{1} = [0; 0; 20; 30];
    P_update{1} = diag([100 100 20 30]);
    
    r = repmat(m_update{1}, 1, duration);
    p = repmat([pi/2; 10; 4], 1, duration);
    Cr = diag([10 10 16 16]);
    Cp = diag([pi/9 .3 .1]);
    
    H = model.H;
    Ar = model.F;
    Ap = eye(3);
    
    Ch = diag([1/4 1/4]);
    Cv = diag([20 8]);
    Cwr = diag([10 10 1 1]);
    Cwp = diag([.05 .001 .001]);

%% Generate measurements
    for i = 1:duration
        num_z(i) = 1 + poissrnd(lambda_z);
        reflection_points = zeros(2, 1);
        h = zeros(num_z(i), 2);
        for j = 1:num_z(i)
            h(j, :) = -1 + 2.* rand(1, 2);
            while norm(h(j, :)) < 1
                h(j, :) = -1 + 2.* rand(1, 2);
    
            end
    
            if rand(1) <= detection_probability
                reflection_points(:, j) = gt_dynamic(1:2, i) + ...
                                        h(j, 1) * gt_shape(2, i) * [cos(gt_shape(1, i)); sin(gt_shape(1, i))] + ...
                                        h(j, 2) * gt_shape(3, i) * [-sin(gt_shape(1, i)); cos(gt_shape(1, i))] + ...
                                        noise_amp * mvnrnd([0; 0], [1 0; 0 1], 1)';
            end
        end
    
        num_c(i) = poissrnd(lambda_c);
        clutters = [unifrnd(clutter_region(1, 1), clutter_region(1, 2), 1, num_c(i));
                unifrnd(clutter_region(2, 1), clutter_region(2, 2), 1, num_c(i))];
        c{i} = clutters;
    
        measurements = [reflection_points clutters];
        z{i} = measurements;
    end
    
    for k = 2:duration
        single_exec_time_start = tic;
%% Predict
        w_birth = .01;
        m_birth = [0; 0; 20; 30];
        P_birth = diag([100 100 20 30]);
    
        [m_predict, P_predict] = predict_KF(model, m_update{k-1}, P_update{k-1});
        w_predict = model.P_S * w_update{k-1};
    
        w_predict = cat(1, w_birth, w_predict);
        m_predict = cat(2, m_birth, m_predict);
        P_predict = cat(3, P_birth, P_predict);
    
%% Update
        w_update{k} = model.P_MD * w_predict;
        m_update{k} = m_predict;
        P_update{k} = P_predict;
        
        [likelihood] = cal_likelihood(z{k}, model, m_predict, P_predict);
    
        if size(z{k}, 2) ~= 0
            [m_temp, P_temp] = update_KF(z{k}, model, m_predict, P_predict);
    
            for i = 1:size(z{k}, 2)
                w_temp = model.P_D * w_predict .* likelihood(:, i);
                w_temp = w_temp ./ (num_c(k) * pdf_c + sum(w_temp));
    
                w_update{k} = cat(1, w_update{k}, w_temp);
                m_update{k} = cat(2, m_update{k}, m_temp(:, :, i));
                P_update{k} = cat(3, P_update{k}, P_temp);
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
            if w_copy(maxIndex) >= 0.5
                indices(i) = maxIndex;
            end
            w_copy(maxIndex) = -inf;
        end
    
        for i = 1:size(indices, 2)
            est{k} = [est{k} m_update{k}(1:2, i)];
        end
    
        est{k} = partition_labeling(d_threshold, est{k});
        num_targets(k) = size(est{k}, 2);
    
        [r(:,k), p(:,k), Cr, Cp] = measurement_update(est{k}, H, r(:,k), p(:,k), Cr, Cp, Ch, Cv);
        
        [r(:,k+1), p(:,k+1), Cr, Cp] = time_update(r(:,k), p(:,k), Cr, Cp, Ar, Ap, Cwr, Cwp);
        
        single_exec_time(k) = toc(single_exec_time_start);
    
    end

%% Evaluation
    disp(['Simulation ', num2str(i_loop), ...
        ':              ', num2str(sum(single_exec_time, 1)), ' (s)']);

    single_ospa = zeros(1, duration);
    ospa_cutoff = 50;
    ospa_order = 1;
    
    if doPlotOSPA
        for i = 1:duration
            [est_mat, gt_mat] = get_uniform_points_boundary([r(1:2, i); p(:, i)]', [gt_dynamic(1:2, i); gt_shape(:, i)]', 50);
            single_ospa(i) = ospa_dist(est_mat, gt_mat, ospa_cutoff, ospa_order);
        end
        
        figure(1);
        hold on;
        
        plot(2:duration, single_ospa(2:end));
        xlabel("Time step");
        ylabel("Distance (in m)");
        title("Multiple OSPA Metrics for Evaluation");
    end

    multi_ospa(i_loop, :) = single_ospa;
    multi_exec_time(i_loop, :) = single_exec_time;

end

%% Multi-evaluation
disp(['-------Total Runtime---------------------Average Runtime---------']);
disp(['         ', num2str(sum(multi_exec_time, 'all')), ...
    ' (s)                       ', num2str(sum(multi_exec_time, 'all')/multi_num_loop), ' (s)']);
average_multi_ospa = sum(multi_ospa, 1) / multi_num_loop;

if doPlotAverageOSPA
    figure (2);
    hold on;

    avg_ospa_plot = plot(2:duration, average_multi_ospa(2:end), '-r', 'LineWidth', 2);

    xlabel("Time step");
    ylabel("Distance (in m)");
    title("Average OSPA Metric for Evaluation");
    legend([avg_ospa_plot], ['For ', num2str(multi_num_loop), ' run-times'], 'Location', 'northeast');
end