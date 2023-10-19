% ------------Single Extended Object Tracking---------
% -----------MATLAB Driving Toolbox Simulation--------
% ====================================================

clc, clear, close all
%% Simulation setting

doSaveGIF = false;
doSaveMP4 = false;

doPlotScenario = false;
doPlotOSPA = false;
doPlotSimpleScenario = false;

isDetectionNoisy = false;
isDetectionProbability = false;

detections_probability = 1;

%% Generate data
[DTB_Data, DTB_Scenario, DTB_Sensor] = DTB_3_Segment_Ideal_Chase_100m;

DTB_Time = vertcat(DTB_Data.Time);
DTB_SamplingTime = DTB_Time(2) - DTB_Time(1);
duration = round(DTB_Time(end) / DTB_SamplingTime + 1);

%% Memory Allocation
z = cell(duration, 1);
d_p = cell(duration, 1);
c = cell(duration, 1);

gt_self_dynamic = zeros(4, duration);
gt_self_orient = zeros(1, duration);
gt_target_dynamic = zeros(4, duration);
gt_target_orient = zeros(1, duration);

num_targets = zeros(duration, 1);
num_detections = zeros(duration, 1);
num_clutters = zeros(duration, 1);
num_reflection_points = zeros(duration, 1);

%% Retrieve ground-truth
for i = 1:duration
    gt_self_dynamic(:, i) = [DTB_Data(i).ActorPoses(1).Position(1:2)'; DTB_Data(i).ActorPoses(1).Velocity(1:2)'];
    gt_self_orient(i) = DTB_Data(i).ActorPoses(1).Yaw;

    gt_target_dynamic(:, i) = [DTB_Data(i).ActorPoses(2).Position(1:2)'; DTB_Data(i).ActorPoses(2).Velocity(1:2)'];
    gt_target_orient(i) = DTB_Data(i).ActorPoses(2).Yaw;
end

%% Retrieve measurements
for i = 1:duration
    num_reflection_points(i) = size(DTB_Data(i).ObjectDetections, 1);
    reflection_points = zeros(2, num_reflection_points(i));
    detections = [];

    for j = 1:num_reflection_points(i)
        reflection_points(:, j) = DTB_Data(i).ObjectDetections{j, 1}.Measurement(1:2);
    end

    meas_pos_sensor = reflection_points;
    meas_conv_angle = gt_self_orient(i) / 180 * pi;
    meas_conv_angle_matrix = [cos(meas_conv_angle) -sin(meas_conv_angle); sin(meas_conv_angle) cos(meas_conv_angle)];
    meas_pos_pole = meas_conv_angle_matrix * meas_pos_sensor + repmat(gt_self_dynamic(1:2, i), 1, num_reflection_points(i));
    reflection_points = meas_pos_pole;
    
    if isDetectionProbability
        for j = 1:num_reflection_points(i)
            if rand(1) <= detections_probability
                detections(:, end + 1) = reflection_points(:, j);
            end
        end
    else
        detections = reflection_points;
    end

    num_detections = size(detections, 2);
    d_p{i} = detections;
end

%% Generate clutters
lambda_c = 50;
clutter_region = [min(gt_self_dynamic(1, :)) - 30, max(gt_self_dynamic(1, :)) + 30;
                min(gt_self_dynamic(2, :)) - 30, max(gt_self_dynamic(2, :)) + 30];
pdf_c = 1 / prod(clutter_region(:, 2) - clutter_region(:, 1));

for i = 1:duration
    num_clutters(i) = poissrnd(lambda_c);

    clutters = [unifrnd(clutter_region(1, 1), clutter_region(1, 2), 1, num_clutters(i)); 
            unifrnd(clutter_region(2, 1), clutter_region(2, 2), 1, num_clutters(i))];
    c{i} = clutters;
    z{i} = [d_p{i} c{i}];
end

%% Predict

%% Update

%% Plot and visualize

if doPlotScenario
    plot(DTB_Scenario);
    pause(2);
    hold on;

    while advance(DTB_Scenario)
        time_step = round(DTB_Scenario.SimulationTime / DTB_SamplingTime) + 1;
        t = time_step;

        if t <= duration
            meas_plot = plot(z{t}(1, :), z{t}(2, :), 'k.', 'MarkerSize', 5);
        end
    end
end

if doPlotSimpleScenario
    figure (2);
    hold on;

    for i = 1:duration
        meas_plot = plot(z{i}(2, :), z{i}(1, :), 'k.', 'MarkerSize', 5);
    end
    gt_self_plot = plot(gt_self_dynamic(2, :), gt_self_dynamic(1, :), '.b-', 'LineWidth', 3, 'MarkerSize', 15);
    gt_target_plot = plot(gt_target_dynamic(2, :), gt_target_dynamic(1, :), '.r-', 'LineWidth', 3, 'MarkerSize', 15);
    
    set(gca, 'XDir', 'reverse');
    xlabel("Position Y");
    ylabel("Position X");
    title("Simple Sensor FOV");
    legend([meas_plot, gt_self_plot, gt_target_plot], 'Measurements', 'Self Vehicle', 'Target Vehicle');
end
%% Evaluation