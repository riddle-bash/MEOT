function indices = topNIndices(vector, n)
    % Kiểm tra nếu n lớn hơn kích thước của vector
    if n > length(vector)
        error('n phải nhỏ hơn hoặc bằng kích thước của vector.');
    end

    % Khởi tạo một mảng lưu trữ các chỉ mục
    indices = zeros(1, n);

    % Lặp qua n lần để tìm n giá trị lớn nhất
    for i = 1:n
        [~, maxIndex] = max(vector);
        indices(i) = maxIndex;

        % Loại bỏ giá trị lớn nhất đã tìm thấy để tìm giá trị lớn tiếp theo
        vector(maxIndex) = -inf; % Sử dụng -inf để đảm bảo không tìm lại giá trị này
    end
end
