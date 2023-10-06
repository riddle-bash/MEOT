function [m_temp,P_temp] = update_single(z,H,R,m,P)
    meas_num = size(z, 2);
    
    for i = 1:meas_num
        % Tính x estimated
        mu = H * m;

        % Tính residual
        residual = z(:,i) - repmat(mu,[1 size(z(:,i),2)]);

        % Tính ma trận Innovation
        S = R + H * P * H';
        Vs = chol(S); 
        inv_sqrt_S = inv(Vs);
        iS = inv_sqrt_S * inv_sqrt_S';
        det_S= prod(diag(Vs))^2;

        % Tính ma trận Kalman Gain
        K = P * H' * iS;

        % Cập nhật trạng thái và ma trận hiệp phương sai
        m_temp = repmat(m,[1 size(z(:,i),2)]) + K * residual;
        P_temp = (eye(size(P)) - K * H) * P;
    end
end