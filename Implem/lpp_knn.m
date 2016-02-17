 function [ W ] = lpp_knn( obs, varargin)
%LPP_KNN Locality preserving projections with standard KNN
% 
%   Optional parameters:
%       'K':        Number of nearest neighbours to use for computing the
%                   discrete Laplacian
%                       Default = 10
%       'distance': Distance to use when finding the KNN. See help of
%                   knnsearch
%                       Default = 'euclidean'
%       'center':   Center the data before processing
%                       Default = true

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants and default values

% Tolerance for detecting convergence and maximum number of iterations
K = 10;
center = true;
knndistance = 'euclidean';

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
            case 'k'
                K = varargin{i+1};
            case 'center'
                center = varargin{i+1};
            case 'distance'
                knndistance = varargin{i+1};
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

% Construct S and D
S = zeros(N,N);

% Find indices of the K nearest neighbours of each element
[Neig, ~] = knnsearch(X,X,'K',K, 'Distance', knndistance);

% Set S(i,j) to 1 if j is a neighbour of i
for i=1:N
    S(i,Neig(i,2:end)) = 1;
end

% Make S symmetric
S = max(S, S');
% Find D
D = diag(sum(S,1));

% Square root of D for convenience
SqD = D^(1/2);

% First eigen analysis
[V,L,~] = svd(SqD*X*(X')*SqD);
U = X' * SqD * V * L^-1;

% Define the data in the new space and perform eigen analysis on the
% Laplacian
Xp = U'*X';
[Q, ~, ~] = svd(Xp*(D-S)*Xp');

% Reverse the vectors, we need the smallest eigen values/vectors first
Ind = 190:-1:1;
Q = Q(:,Ind);

% Final transformation matrix
W = U*Q;

 end
