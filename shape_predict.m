function [p,Cp]= shape_predict(p,Cp,Ap,Cwp)
    p = Ap*p;
    Cp = Ap*Cp*Ap'+Cwp;
end