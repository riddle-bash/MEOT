function [m_update, P_update] = update_KF(z, model, m, P)

    plength= size(m,2);
    zlength= size(z,2);

    m_update = zeros(model.xdim,plength,zlength);
    P_update = zeros(model.xdim,model.xdim,plength);
    for idx = 1:plength
        [m_temp, P_temp] = update_single(z,model.H,model.R,m(:,idx),P(:,:,idx));
        m_update(:,idx,:) = m_temp;
        P_update(:,:,idx) = P_temp;
    end
end

function [m_temp,P_temp] = update_single(z,H,R,m,P)
    % Tính x estimated
    mu = H * m;

    % Tính residual
    residual = z - repmat(mu,[1 size(z,2)]);

    % Tính ma trận Innovation
    S = R + H * P * H';
    Vs = chol(S); 
    inv_sqrt_S = inv(Vs);
    iS = inv_sqrt_S * inv_sqrt_S';
    det_S= prod(diag(Vs))^2;

    % Tính ma trận Kalman Gain
    K = P * H' * iS;

    % Cập nhật trạng thái và ma trận hiệp phương sai
    m_temp = repmat(m,[1 size(z,2)]) + K * residual;
    P_temp = (eye(size(P)) - K * H) * P;
end