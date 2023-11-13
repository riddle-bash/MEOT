function [m_predict, P_predict] = predict_KF(model, m, P)
    % x: trạng thái trước khi update
    % P: ma trận hiệp phương sai trạng thái trước khi update
    % F: ma trận chuyển đổi trạng thái
    % Q: ma trận hiệp phương sai của nhiễu quá trình

    plength= size(m,2);

    m_predict = zeros(size(m));
    P_predict = zeros(size(P));

    for idx = 1:plength
        [m_temp,P_temp] = predict_single(model.F, model.Q, m(:,idx), P(:,:,idx));
        m_predict(:,idx) = m_temp;
        P_predict(:,:,idx) = P_temp;
    end
end

function [m_predict,P_predict] = predict_single(F,Q,m,P)
    % Dự đoán trạng thái mới
    m_predict = F * m;

    % Dự đoán ma trận hiệp phương sai mới
    P_predict = Q + F * P * F';
end