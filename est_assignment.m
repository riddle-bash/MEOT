function [est_m, gt, dim] = est_assignment(est_m, gt)
    m = size(gt, 2);
    n = size(est_m, 2);
    for est_idx = 1:n
        for gt_idx = 1:m
            % Calculate distance from est_m to gt
            assign_tmp(est_idx, gt_idx) = cal_dist(gt{gt_idx}(1:2), est_m{est_idx}(1:2));
        end
    end
    if m < n
        dim = n;
        for gt_idx = 1:m
            [~, sortOrder(gt_idx)] = min(assign_tmp(:, gt_idx));
        end
        sortOrder = [sortOrder setdiff(1:n,sortOrder)];
        est_m = est_m(sortOrder);
        gt{dim} = {};
    else
        dim = m;
        for est_idx = 1:n
            [~, sortOrder(est_idx)] = min(assign_tmp(est_idx, :));
        end
        sortOrder = [sortOrder setdiff(1:m,sortOrder)];
        gt = gt(sortOrder);
        if n ~= m
            est_m{dim} = {};
        end
    end
end

function dist = cal_dist(a, b)
    dist = 0;
    for i = 1:size(a, 2)
        dist = dist + sqrt((a(1, i) - b(1))^2 + (a(2, i) - b(2))^2);
    end
    dist = dist / size(a, 2);
end