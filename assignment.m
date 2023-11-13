function [Meas_tmp, est_m, p] = assignment(Meas, est_m, p, num_target)
% Function to propagate cell
% Input: Meas - measurement set
%        est_m - kinematic state set
%        p    - shape state set
%        num_target - number of target
% Output: propagated Meas_tmp, est_m, p

    n = size(Meas, 2);
    m = size(est_m, 2);
    assign_tmp = zeros(n, m);
    for i = 1:n
        for j = 1:m
            % Calculate distance from est_m to cell W
            assign_tmp(i, j) = cal_dist(Meas{i}.Meas, est_m(1:2, j));
        end
    end
    for i = 1:num_target
        % The cell with minimum distance from est_m is chosen 
        % with sortOrder index
        [~, sortOrder] = min(assign_tmp(:, i));
        % Save cell matrix propogate to element est_m ith
        Meas_tmp{i} = Meas{sortOrder}.Meas;
    end
end

function dist = cal_dist(a, b)
    dist = 0;
    for i = 1:size(a, 2)
        dist = dist + sqrt((a(1, i) - b(1))^2 + (a(2, i) - b(2))^2);
    end
    dist = dist / size(a, 2);
end