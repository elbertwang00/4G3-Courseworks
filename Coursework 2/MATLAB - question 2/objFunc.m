function [L, dL] = objFunc(params, X)
    logA = params(:,1:width(X));
    logb = params(:,width(X)+1);

    A = exp(logA);
    A(logical(eye(size(A)))) = 0; % Set the diagonal to 0 so that each k is dependent only on the other neurons
    b = exp(logb);
    XT2 = (X').^2; % Element-wise square of transpose of X;

    varK = A * XT2; % Sum part of variance with akj elements
    for n=1:width(varK) % Adding bk's
        varK(:,n) = varK(:,n) + b;
    end
    
    const = -0.5*log(2*pi);
    logp = const - 0.5*(log(varK) + XT2./varK); % Conditional prob.
    L = -sum(logp, 'all'); % Likelihood function (negate for minimization problem)
    
    oneMinusRatioNorm = -0.5./varK .* (1 - XT2./varK);
    dLdb = -sum(oneMinusRatioNorm, 2); % Partial derivatives wrt bk
    dLdA = zeros(size(A));
    
    for k=1:height(A)
        for j=1:width(A)
            if k==j % Set cases when k=j to 0 (diagonal of partial derivative)
                dLdA(k,j) = 0;
            else % Partial derivatives wrt akj
                dLdA(k,j) = -sum(XT2(j,:) .* oneMinusRatioNorm(k,:));
            end
        end
    end

    dL = [dLdA.*A, dLdb.*b]; % Concatenate dLdA and dLdb for output
end