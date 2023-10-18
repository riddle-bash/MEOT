% -------------Single Extended Object Tracking----------------
% ---Using Gaussian Mixture Probability Hypothesis Density----
% ------------------------------------------------------------
% ------------------------------------------------------------

clc, clear, close all
%% Simulation setting
duration = 100;
model = gen_model;

%% Ground-truth, noise setting

gt1(:,1) = [0; 0; 20; 30];
gt1_shape = [56.3/180 * pi; 2 * model.carSize];

for i = 2:duration
    gt1(:,i) = model.F * gt1(:,i-1);
end

%% Generate measurement

z = cell(duration, 1);
c = cell(duration, 1);
num_z = zeros(duration, 1);
num_c = zeros(duration, 1);
lambda_z = 5;
lambda_c = 50;
noise_amp = 5;

clutter_region = [gt1(1,1) - 10, gt1(1,end) + 10;
    gt1(2,1) - 10, gt1(2,end) + 10];
pdf_c = 1/prod(clutter_region(:,2) - clutter_region(:,1));

for i = 1:duration
    num_z(i) = poissrnd(lambda_z);
    z1 = zeros(2, num_z(i));
    for j = 1:num_z(i)
        h(j, :) = -1 + 2.* rand(1, 2);
        while norm(h(j, :)) < 1
            h(j, :) = -1 + 2.* rand(1,2);
        end

        if rand(1) <= model.P_D
        end

        z1(:, j) = gt1(1:2, i) + ...
            h(j, 1) * gt1_shape(2) * [cos(gt1_shape(1)); sin(gt1_shape(1))] + ...
            h(j, 2) * gt1_shape(3) * [-sin(gt1_shape(1)); cos(gt1_shape(1))] + ...
            noise_amp * mvnrnd([0; 0], [1 0; 0 1], 1)';
            
    end
    
    num_c(i) = poissrnd(lambda_c);
    c{i} = [unifrnd(clutter_region(1,1),clutter_region(1,2),1,num_c(i)); unifrnd(clutter_region(2,1),clutter_region(2,2),1,num_c(i))];

    z{i} = [z1 c{i}];
end

%% Prior
w_update = cell(1,duration);
m_update = cell(1,duration);
P_update = cell(1,duration);

w_update{1} = .01;
m_update{1} = [0; 0; 20; 30];
P_update{1} = diag([100 100 20 30]);

est = cell(1, duration);
num_targets = zeros(1, duration);
exec_time = zeros(1, duration);

elim_threshold = 1e-5;
merge_threshold = 4;
L_max = 100;
d_threshold = 50;

r = repmat(m_update{1}, 1, duration);
p = repmat([pi/4 10 4]', 1, duration);
Cr = diag([10 10 16 16]);
Cp = diag([pi/9 .3 .1]);

H = [1 0 0 0; 0 1 0 0];
Ar = model.F;
Ap = eye(3);

Ch = diag([1/4 1/4]);
Cv = diag([20 8]);
Cwr = diag([10 10 1 1]);
Cwp = diag([.05 .001 .001]);

for k = 2:duration
    execution = tic;
%% Predict
    w_birth = .01;
    m_birth = [0; 0; 20; 20];
    P_birth = diag([100 100 20 20]);

    [m_predict, P_predict] = predict_KF(model, m_update{k-1}, P_update{k-1});
    w_predict = model.P_S * w_update{k-1};

    w_predict = cat(1, w_birth, w_predict);
    m_predict = cat(2, m_birth, m_predict);
    P_predict = cat(3, P_birth, P_predict);

%% Update
    w_update{k} = model.P_MD * w_predict;
    m_update{k} = m_predict;
    P_update{k} = P_predict;

    [likelihood_tmp] = cal_likelihood(z{k}, model, m_predict, P_predict);

    if size(z{k}, 2) ~= 0
        [m_temp, P_temp] = update_KF(z{k}, model, m_predict, P_predict);
        
        for i = 1:size(z{k}, 2)
            w_temp = model.P_D * w_predict .* likelihood_tmp(:,i);
            w_temp = w_temp ./ (num_c(k) * pdf_c + sum(w_temp));

            w_update{k} = cat(1, w_update{k}, w_temp);
            m_update{k} = cat(2, m_update{k}, m_temp(:,:,i));
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
        if w_copy(maxIndex) > 0.5
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
    
    exec_time(k) = toc(execution);

    disp(['Time step ', num2str(k), ...
        ': measurements = ', num2str(size(z{k}, 2)), ...
        ', estimations = ', num2str(num_targets(k)), ...
        ', reflection points = ', num2str(num_z(k)), ...
        ', clutters = ', num2str(num_c(k)), ...
        ', execution time = ', num2str(exec_time(k)) , 's']);

end

%% Plot and visualize
figure (1);
hold on;
gt_plot = plot(gt1(1,:), gt1(2,:), '-r.', 'LineWidth', 1.5, 'MarkerSize', 15);

for t = 1:duration
    meas_plot = plot(z{t}(1,:), z{t}(2,:), 'k.', 'MarkerSize', 1);
end

xlim([clutter_region(1,1) clutter_region(1,2)]);
ylim([clutter_region(2,1) clutter_region(2,2)]);
xlabel('Position X');
ylabel('Position Y');
title('Sensor FOV');
legend([gt_plot, meas_plot], 'Ground-truth', 'Measurements', 'Location', 'southeast');

figure (2);
hold on
gt_plot = plot(gt1(1,:), gt1(2,:), '-r.', 'LineWidth', 1.5, 'MarkerSize', 15);

for t = 1:duration
    for k = 1:num_targets(t)
        est_plot = plot(est{t}(1, k), est{t}(2, k), 'b*');
    end

    c_plot = plot(c{t}(1, :), c{t}(2, :), 'k.', 'MarkerSize', 1);
end

xlim([clutter_region(1,1) clutter_region(1,2)]);
ylim([clutter_region(2,1) clutter_region(2,2)]);
xlabel('Position X');
ylabel('Position Y');
title('GM-PHD Estimation');
legend([gt_plot, est_plot, c_plot], 'Ground-truth', 'Estimation', 'Clutters', 'Location', 'southeast');

figure (3);
hold on;

for t = 1:duration
    if mod(t, 1) == 0
        gt_center_plot = plot(gt1(1,t), gt1(2,t), 'r.');
        gt_plot = plot_extent([gt1(1:2,t); gt1_shape], '-', 'r', 1);

        est_center_plot = plot(r(1,t), r(2,t), 'b+');
        est_plot = plot_extent([r(1:2,t); p(:,t)], '-', 'b', 1);
    end
end

xlim([clutter_region(1,1) 400]);
ylim([clutter_region(2,1) 600]);
xlabel('Position X');
ylabel('Position Y');
title('Extended GM-PHD Estimation');
legend([gt_plot, est_plot], 'Ground-truth', 'Estimation', 'Location', 'southeast');

%% Evaluation
doPlotOSPA = false;
disp(['------------------------------Total run time: ', num2str(sum(exec_time, 2)), 's---------------------------']);

if doPlotOSPA
    ospa = zeros(1, duration);
    ospa_cutoff = 50;
    ospa_order = 1;
    
    for t = 2:duration
        [gt_mat, est_mat] = get_uniform_points_boundary([gt1(1:2,t); gt1_shape]', [r(1:2,t); p(:,t)]', 50);
        ospa(t) = ospa_dist(gt_mat, est_mat, ospa_cutoff, ospa_order);
    end

    figure (4);
    hold on;
    
    plot(2:duration, ospa(2:end));

    xlabel('Time step');
    ylabel('Distance (in m)');
    title('OSPA Evaluation');
end