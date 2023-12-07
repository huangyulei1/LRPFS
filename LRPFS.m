function [X_new, W, V, iter, obj, idx] = LRPFS(X, Y, D, alpha, lamda, p, maxIter)
% LRPFS: Unsupervised feature selection with latent relationship penalty term.
% 
% Input:
%   X       - Data matrix (d*n). Each column vector of data is a sample vector.
%   Y       - The labels of the data (n*1).
%   D       - Score matrix (n*n).
%   alpha   - The sparse constraint parameter.
%   lamda   - The scale parameter.
%   k       - Generate the parameters of the score matrix D, k is equal to 0 or 5..
%   p       - The number of featuers to be selected
%   maxIter - The maximum number of iterations
%
% Output:
%   X_new   - The selected subset of features (n*p).
%--------------------------------------------------------------------------
%    Examples:
%       load('COIL20.mat');
%       maxIter=30;p=90; alpha=1e3; lamda=1e-1; k=5;% k is equal to 0 or 5.
%       options.NeighborMode='KNN'; options.k=k; 
%       options.t=1e+1; options.WeightMode='Heatkernel';
%       S=lapgraph(X,options);
%       D=diag(sum(S));
%       [X_new,W,V,iter,obj]=LRPFS(fea',gnd,D,alpha,lamda,p,maxIter);
%       %The features in X_new will be sorted based on their importance.
rng('default');
C=length(unique(Y));
[m,~]=size(X);
eta=0.1;

[~,V]=litekmeans(X,C);V=V'; 

U=eye(m);

W=(X*X'+eta*eye(m))\(X*V);
W=max(W,1e-8);
X=X';
for iter=1:maxIter
 
    W=W.*((X'*V)./(X'*X*W+alpha*U*W+eps));
    V=V.*((X*W+ 2*lamda*D*X*X'*D'*V)./(V+ 2*V*V'*V+eps));
    
    Wi=sqrt(sum(W.*W,2)+eps);
    u=0.5./Wi;
    U=diag(u);

    
    obj(iter)=trace((X*W-V)*(X*W-V)')+trace((V*V'-lamda*D*X*X'*D')*(V*V'-lamda*D*X*X'*D')')+alpha*sum(Wi);

    if iter>2
        if abs(obj(iter)-obj(iter-1))/obj(iter-1)<1e-6
            break
        end
    end
end

score=sqrt(sum(W.*W,2));
[~, idx]=sort(score,'descend');
X_new=X (:,idx(1:p));

end