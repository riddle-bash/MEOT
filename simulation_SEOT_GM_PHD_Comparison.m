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

clc, clear, close all
%% Multi-simulation setting
multi_num_loop = 1;
multi_simu_dur = 20;

custom_scenario = 1;

doPlotOSPA = false;
doPlot_SensorFOV = true; 
doPlot_Estimation = false;
doPlot_ExtendedEstimation = false;

doPlotExtended_SensorFOV = true;
doPlotExtended_Estimation = false;
doPlotExtended_ExtendedEstimation = false;

model = gen_model;
duration = multi_simu_dur;
scenarioID = custom_scenario;

%% Ground-truth and noise setting
[gt_dynamic, gt_orient] = ground_truth_generate(scenarioID, duration, model.F, false);
gt_shape = [gt_orient; repmat(model.carSize, 1, duration)];

%% Generate measurements
z = cell(duration, 1);
c = cell(duration, 1);
num_z = zeros(duration, 1);
num_c = zeros(duration, 1);
lambda_z = 5;
lambda_c = 50;
noise_amp = 1;

clutter_region = [gt_dynamic(1, 1) - 10  gt_dynamic(1, end) + 10;
                    gt_dynamic(2, 1) - 10 gt_dynamic(2, end) + 10];
pdf_c = 1/prod(clutter_region(:, 2) - clutter_region(:, 1));

for i = 1:duration
    num_z(i) = poissrnd(lambda_z);
    reflection_points = zeros(2, 1);
    h = zeros(num_z(i), 2);
    for j = 1:num_z(i)
        h(j, :) = -1 + 2.* rand(1, 2);
        while norm(h(j, :)) < 1
            h(j, :) = -1 + 2.* rand(1, 2);

        end

        if rand(1) <= model.P_D
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
                
%% Prior setting



%% Predict



%% Update



%% Plot and visualize
if doPlot_SensorFOV
    figure(1);
    hold on;
    
    if doPlotExtended_SensorFOV
        for i = 1:duration
            gt_center_plot = plot(gt_dynamic(1, i), gt_dynamic(2, i), 'r.', 'MarkerSize', 15);
            gt_plot = plot_extent([gt_dynamic(1:2, i); gt_shape(:, i)], '-', 'r', 3);

            meas_plot = plot(z{i}(1, :), z{i}(2, :), 'k.', 'MarkerSize', 5);
        end
    else
        gt_plot = plot(gt_dynamic(1, :), gt_dynamic(2, :), '-r.', 'LineWidth', 1.5, 'MarkerSize', 15);
        for i = 1:duration
            meas_plot = plot(z{i}(1, :), z{i}(2, :), 'k.', 'MarkerSize', 5);
        end
    end

    xlim([clutter_region(1,1) clutter_region(1,2)]);
    ylim([clutter_region(2,1) clutter_region(2,2)]);
    xlabel('Position X');
    ylabel('Position Y');
    title('Sensor FOV');
    legend([gt_plot, meas_plot], 'Ground-truth', 'Measurements', 'Location', 'southeast');
end


%% Evaluation
