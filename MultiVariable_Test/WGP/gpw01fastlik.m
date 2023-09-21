function [fX, dfX] = gpw01fastlik(X, input, t, dflag);

global invQ logdetQ; % get from gpw01lik to avoid recomputation

% gpw01fastlik: Compute minus log likelihood and its derivatives with respect to
% warping parameters only, for the warped Gaussian process with Gaussian
% covariance. Return zero gradients for covariance hyperparameters.
%
% Based on code for GPs by Carl Edward Rasmussen (gp01lik)
% 
% X      is the vector of parameters of length D+2+3*num, where num is
%        the number of tanh functions in the warp (see warp):
%
% X = [ log(w_1)
%       log(w_2) 
%        .
%       log(w_D)
%       log(v1)
%       log(v0) 
%       a
%       b
%       c ]
%
% w,v    are the hyperparameters of the covariance (see gp01lik)
% a,b,c  are vectors of warping parameters of length num (see warp)
% 
% input  is an (n x D) array of data
% t      is a (n x 1) vector of target values
% dflag  == 'g' outputs only gradients; dflag = 'f' outputs only
%        function values
% 
% Use in conjunction with gradient based minimizer such as Carl
% Edward Rasmussen's "minimize":
%
% [X, fX, i] = minimize(X, 'gpw01fastlik', length, input, t, dflag)

if nargin < 4, dflag = 'default'; end
[n, D] = size(input);       % number of examples and dimension of input space
num = (length(X) - D - 2)/3; % number of tanh functions
expX = exp(X);

% prior means and variances in hyperparameters
% alter these to do MAP rather than ML optimization
sX = repmat(1e15,length(X),1);
muX = zeros(size(X));
%sX(D+3+num:D+2+2*num) = 4; % may want to stop warp becoming too steep

for i = 1:num % the parameters for warping function
    ea(i,1) = expX(D+2+i); eb(i,1) = expX(D+2+num+i);
    c(i,1) = X(D+2+2*num+i);
end

z = warp(t,ea,eb,c);   % warp target values onto latent variable z

invQz = invQ*z;
w = ones(n,1);
for i = 1:num % various useful constructs
    s{i} = eb(i)*(t + c(i));
    r{i} = tanh(s{i});
    y{i} = 1 - r{i}.^2;
    w = w + ea(i)*eb(i).*y{i};
end

% calculate minus log likelihood ...

if strcmp(dflag, 'default') || strcmp(dflag, 'f')
  jacob = sum(log(w)); % jacobian term 
  fX = 0.5*logdetQ + 0.5*z'*invQz + 0.5*n*log(2*pi) - jacob + ...
       0.5*sum(((X-muX).^2)./sX);% + log(2*pi*sX));
else
   fX = [];
end

% ... and its partial derivatives

if strcmp(dflag, 'default') || strcmp(dflag, 'g')
  dfX = zeros(D+2+3*num,1);  % set the size of the derivative vector

  for i = 1:num % derivatives wrt warping parameters
    dfX(D+2+i) = ea(i)*(r{i}'*invQz  - sum(eb(i).*y{i}./w));
    dfX(D+2+num+i) = eb(i)*((ea(i)*(t+c(i)).*y{i})'*invQz - sum((ea(i).*y{i} - ea(i)*eb(i)*2.*y{i}.*r{i}.*(t+c(i)))./w));
    dfX(D+2+2*num+i) = (ea(i)*eb(i).*y{i})'*invQz + sum(ea(i)*eb(i)^2*2.*y{i}.*r{i}./w);
  end
  for d = D+3:length(X)
    dfX(d) = dfX(d) + (X(d)-muX(d))/sX(d);
  end
else
  dfX = [];
end