function [P] = distance_partitioning(d_threshold, Meas)
    P = struct();
    % Check if Meas has more than 1 measurement inside
    if size(Meas, 2) ~= 1
        cellNumber = partitioning(d_threshold, Meas);
        count = zeros(1, max(cellNumber));
        label = [];
    
        for i = 1:max(cellNumber)
            P.W{i}.Meas = Meas(:, cellNumber == i);
        end
    
%         while(max(count) ~= -Inf)
%             label = [];
%             [~, mostAppear] = max(count);
%             for i = 1:size(cellNumber, 1)
%                 if cellNumber(i) == mostAppear
%                     label(end + 1) = i;
%                 end
%             end
%             labeled_est{end + 1} = Meas(:, label);
%             count(count == max(count)) = -Inf;
%         end
    else
        P.W{1} = Meas;
    end
end

function [cellNumber] = partitioning(d_threshold, m_update)
    cellNumber = zeros(size(m_update,2),1);
    cellID = 1;

    for i = 1:size(m_update, 2)
        if cellNumber(i) == 0
            cellNumber(i) = cellID;
            cellNumber = findNeighbor(i, cellNumber, cellID, m_update, d_threshold);
            cellID = cellID + 1;
        end
    end
end

function cellNumber = findNeighbor(i, cellNumber, cellID, m_update, d_threshold)
    for j = 1:size(m_update, 2)
        if (j ~= i) && (d_distance(m_update(:,i), m_update(:,j)) < d_threshold) && cellNumber(j) == 0
            cellNumber(j) = cellID;
            cellNumber = findNeighbor(j, cellNumber, cellID, m_update, d_threshold);
        end
    end
end

function d = d_distance(a, b)
    d = sqrt((a(1)-b(1))*(a(1)-b(1))+(a(2)-b(2))*(a(2)-b(2)));
end


