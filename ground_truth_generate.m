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
            gt(:, 1) = [0; 0; 20; 30];
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
            
        case 3
            gt(:, 1) = [0; 0; 20; 30];
            orient(:, 1) = atan(gt(4, 1) / gt(3, 1));
            for i = 2:gtDuration
                if i <= gtDuration / 6
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 1 / 6 && i <= gtDuration * 2 / 6
                    gt(3:4, i-1) = [20; 0];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 2 / 6 && i <= gtDuration * 3 / 6
                    gt(3:4, i-1) = [20; -30];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 3 / 6 && i <= gtDuration * 4 / 6
                    gt(3:4, i-1) = [-20; -30];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 4 / 6 && i <= gtDuration * 5 / 6
                    gt(3:4, i-1) = [-20; 0];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 5 / 6 && i <= gtDuration * 6 / 6
                    gt(3:4, i-1) = [-20; 30];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                end
            end
            
        case 4
            v_h = 10*sqrt(2)/acos(15/180*pi);
            v_x = 10*sqrt(2)/acos(15/180*pi) * sin(30/180*pi);
            v_y = 10*sqrt(2)/acos(15/180*pi) * cos(30/180*pi);
            gt(:, 1) = [0; 0; v_x; v_y];
            orient(:, 1) = atan(gt(4, 1) / gt(3, 1));
            for i = 2:gtDuration
                if i <= gtDuration / 12
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 1 / 12 && i <= gtDuration * 2 / 12
                    gt(3:4, i-1) = [v_y; v_x];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 2 / 12 && i <= gtDuration * 3 / 12
                    gt(3:4, i-1) = [v_h; 0];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 3 / 12 && i <= gtDuration * 4 / 12
                    gt(3:4, i-1) = [v_y; -v_x];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 4 / 12 && i <= gtDuration * 5 / 12
                    gt(3:4, i-1) = [v_x; -v_y];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 5 / 12 && i <= gtDuration * 6 / 12
                    gt(3:4, i-1) = [0; -v_h];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 6 / 12 && i <= gtDuration * 7 / 12
                    gt(3:4, i-1) = [-v_x; -v_y];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 7 / 12 && i <= gtDuration * 8 / 12
                    gt(3:4, i-1) = [-v_y; -v_x];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 8 / 12 && i <= gtDuration * 9 / 12
                    gt(3:4, i-1) = [-v_h; 0];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 9 / 12 && i <= gtDuration * 10 / 12
                    gt(3:4, i-1) = [-v_y; v_x];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 10 / 12 && i <= gtDuration * 11 / 12
                    gt(3:4, i-1) = [-v_x; v_y];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                elseif i > gtDuration * 11 / 12 && i <= gtDuration * 12 / 12
                    gt(3:4, i-1) = [0; v_h];
                    gt(:, i) = F * gt(:, i-1) + gt_noise(:, i);
                end
                orient(:, i) = atan(gt(4, i)/gt(3, i));
            end

        otherwise
            disp("Incorrect ground-truth input.");
    end
    
    ground_truth = gt;
end

