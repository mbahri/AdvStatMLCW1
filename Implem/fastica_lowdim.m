 function [ W ] = fastica_lowdim( obs, varargin )
%FASTICA Implements the FastICA algorithm on a low-dimensional subspace
% 
%   Optional parameters:
%       'maxiter':   Maximum number of iterations.
%                       Default = 500
%       'tolerance': Thereshold below which two iterations are considered
%                    identical.
%                       Default = 1e-4
%       'nic':       Number of independent components to compute.
%                       Default = number of low-dimensional features

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants and default values

% Tolerance for detecting convergence and maximum number of iterations
e = 1e-4;
MAXITER = 500;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pre-treatments

% Data whitening using PCA
whiteningMatrix = pcomp(obs, 'whiten', true);
X = bsxfun(@minus, obs, mean(obs));	% Centering
X = (X*whiteningMatrix)';           % Whitened low-dimensional data

% Quantities of interest:
N = size(X, 2);     % Number of observations
F = size(X, 1);     % Number of features
NIC = F;            % Number of independent components to compute

% Vector of ones
OO = ones(N,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read the optional parameters

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        if ~ischar (varargin{i}),
            error (['Unknown type of optional parameter name (parameter' ...
                ' names must be strings).']);
        end
        % change the value of parameter
        switch lower (varargin{i})
            case 'maxiter'
                MAXITER = varargin{i+1};
            case 'tolerance'
                e = varargin{i+1};
            case 'nic'
                NIC = varargin{i+1};
                if NIC > F
                    error('Error: cannot compute more independent components than features!');
                end
                if NIC < 1
                    error('Error: cannot compute less than 1 independent component!');
                end
            otherwise
                error(['Unrecognized parameter: ''' varargin{i} '''']);
        end;
    end;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Actual implementation

% Start with normally distributed random orthonormal vectors
W = orth(randn(F,NIC));

for i = 1:NIC
    % Counter for the number of iterations
    k = 0;
    
    fprintf('%d ', i);

    % ith independent component
    WW = W(:,i);
    % Copy to allow comparison between two iterations
    WWp = zeros(F,1);

    % Loop until WW doesn't change from an iteration to the next
    while norm(WW - WWp) > e && k < MAXITER
        k = k + 1;
        % Update the copy
        WWp = WW;

        % Avoids computing this product twice
        Y = WW'*X;
        % Newton step using y -> y.^4
        WW = (X*(Y.^4)' - (4*Y.^3)*OO*WW)/N;

        % Gram-Schmidt step
        WW = WW - sum(W(:,1:i-1)*diag(W(:,1:i-1)'*WW), 2);

        % Normalize WW
        WW = WW/sqrt(WW'*WW);
    end
    if k >= MAXITER
        fprintf('\n[IC %d] - Max number of iterations reached before convergence!\n', i);
    end

    % Update the matrix
    W(:,i) = WW;
end
fprintf('FastICA terminated.\n');

% The low-dimensional ICs computed on the whitened data need to be
% expressed in the original basis
W = (W * whiteningMatrix(:,1:NIC)')';

 end
