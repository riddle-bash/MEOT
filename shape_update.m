function [p, Cp] = shape_update(y,H,r,p,Cr,Cp,Ch,Cv)
nk = size(y,2); % number of measurements at time k

%yep they still run thru each measurement and update each of them
for i = 1:nk
    
    [CI,CII,M,F,Ftilde] = get_auxiliary_variables(p,Cp,Ch);

    yi = y(:,i);
    
    % calculate moments for the kinematic state update
    yibar = H*r;
    Cy = H*Cr*H'+CI+CII+Cv;
    
    % construct pseudo-measurement for the shape update
    Yi = F*kron(yi-yibar,yi-yibar); 
    % calculate moments for the shape update 
    Yibar = F*reshape(Cy,[4,1]);
    CpY = Cp*M';
    CY = F*kron(Cy,Cy)*(F + Ftilde)';
    % update shape 
    p = p + CpY*CY^(-1)*(Yi-Yibar);
    Cp = Cp - CpY*CY^(-1)*CpY';
    % Enforce symmetry of the covariance
    Cp = (Cp+Cp')/2;
end
end


function [CI,CII,M,F,Ftilde] = get_auxiliary_variables(p,Cp,Ch)
alpha = p(1);
l1 = p(2);
l2 = p(3);

S = [cos(alpha) -sin(alpha); sin(alpha) cos(alpha)]*diag([l1 l2]);
S1 = S(1,:);
S2 = S(2,:);

J1 = [-l1*sin(alpha) cos(alpha) 0; -l2*cos(alpha) 0 -sin(alpha)];
J2 = [ l1*cos(alpha) sin(alpha) 0; -l2*sin(alpha) 0  cos(alpha)];

CI = S*Ch*S';
CII(1,1) = trace(Cp*J1'*Ch*J1);
CII(1,2) = trace(Cp*J2'*Ch*J1);
CII(2,1) = trace(Cp*J1'*Ch*J2);
CII(2,2) = trace(Cp*J2'*Ch*J2);

M = [2*S1*Ch*J1; 2*S2*Ch*J2; S1*Ch*J2 + S2*Ch*J1];

F = [1 0 0 0; 0 0 0 1; 0 1 0 0];
Ftilde = [1 0 0 0; 0 0 0 1; 0 0 1 0];
end