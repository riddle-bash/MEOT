function [likelihood] = cal_likelihood(z, model, m, P)
    plength= size(m,2);
    zlength= size(z,2);

    l_threshold = 1e-300;
    likelihood= zeros(plength,zlength);
    for idx = 1:plength
        % Tính x estimated
        mu = model.H * m(:,idx);

        % Tính ma trận Innovation
        S = model.R + model.H * P(:,:,idx) * model.H';
        Vs = chol(S);
        inv_sqrt_S = inv(Vs);
        iS = inv_sqrt_S * inv_sqrt_S';
        det_S= prod(diag(Vs))^2;
        % Tính likelihood của phép đo
        % pdf = [(2*pi)^(-N/2)*det(S)^(-1/2)] * exp(-1/2 * (z-m)' * 1/S * (z-m))
        % log(pdf) = -0.5*N*log(2*pi) - 0.5*log(det(S)) - (0.5)*(z-m)'*1/S*(z-m)
        % exp(log(pdf)) = pdf
        likelihood(idx,:) = exp(-0.5*size(z,1)*log(2*pi) - 0.5*log(det_S) - ...
            0.5*dot((z-repmat(mu,[1 size(z,2)])),iS*(z-repmat(mu,[1 size(z,2)]))))';
    end
    likelihood(likelihood < l_threshold) = l_threshold;
end