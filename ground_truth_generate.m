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
% -------------------------H. Hai Nam- 2023--------------------------

function [ground_truth, orient] = ground_truth_generate(gtID,gtDuration,gtDynamicFunc,isNoisy)
    F = gtDynamicFunc;
    gt = zeros(4, gtDuration);
    orient = zeros(1, gtDuration);
    gt_noise = zeros(4, gtDuration);

    if isNoisy
        gt_noise = mvnrnd(zeros(4,1), eye(4), gtDuration)';
    end

    switch gtID
        case 1
            gt(:, 1) = [0; 0; 20; 30];
            orient(:, 1) = atan(gt(4, 1)/gt(3, 1));
            for i = 2:gtDuration
                gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                orient(:, i) = atan(gt(4, i)/gt(3, i));
            end

        case 2
            gt(:, 1) = [100; 100; 20; 30];
            orient(:, 1) = atan(gt(4, 1)/gt(3, 1));
            for i = 2:gtDuration/3
                gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                orient(:, i) = atan(gt(4, i)/gt(3, i));
            end

            currentTime = floor(gtDuration/3);
            gt(:, currentTime-1) = [gt(1:2, currentTime-1); 20; -30];
            timeLeft = gtDuration - gtDuration/3 + 1;
            for i = currentTime:(currentTime + timeLeft)
                gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                orient(:, i) = atan(gt(4, i)/gt(3, i));
            end
            
        otherwise
            disp("Incorrect ground-truth input.");
    end
    
    ground_truth = gt;
end

