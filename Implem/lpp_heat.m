 function [ W ] = lpp_heat( obs, varargin)
%LPP_HEAT Locality preserving projections with heat kernel
% 
%   Optional parameters:
%       'T':        T factor in the heat kernel
%                       Default = 1e7
%       'center':   Center the data before processing
%                       Default = true

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants and default values

% Tolerance for detecting convergence and maximum number of iterations
T = 1e7;
center = true;

% Quantities of interest:
N = size(obs, 1);      % Number of observations

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
            case 't'
                T = varargin{i+1};
            case 'center'
                center = varargin{i+1};
            otherwise
                error(['Unrecognized parameter: ''' varargin{i} '''']);
        end;
    end;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pre-treatments

if center
    % Mean-centered data
    X = bsxfun(@minus, obs, mean(obs));
else
    fprintf('Data will not be centered.\n');
    X = obs;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Actual implementation

% Distances squared
DotProd = X*X';
NormsSq = repmat(diag(DotProd),1,N);

Dist = NormsSq + NormsSq' - 2 * DotProd;

% Construct S with heat kernel and D by summing
S = exp(- Dist / T) - eye(N,N); % Remove the ones on the diagonal
D = diag(sum(S,1));             % Find D

% Square root of D
SqD = D^(1/2);

% First eigen analysis
[V,L,~] = svd(SqD*X*(X')*SqD);
U = X' * SqD * V * L^-1;

% Define the data in the new space and perform eigen analysis on the
% Laplacian
Xp = U'*X';
[Q, ~, ~] = svd(Xp*(D-S)*Xp');

% Reverse the vectors, we need the smallest eigen values/vectors first
Ind = size(Q,2):-1:1;
Q = Q(:,Ind);

% Final transformation matrix
W = U*Q;

 end
