function [ basis ] = pcomp( data, varargin )
%PCOMP Performs PCA on the data
%   Calls pcomp_std or pcomp_trick depending on the shape of the data
%
%   Optional parameters:
%       'whiten': {true, false} indicates if whitening is needed
%                   Default = false

% Default value for whitening
scale = false;

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
     case 'whiten'
      scale = varargin{i+1};
     otherwise
      error(['Unrecognized parameter: ''' varargin{i} '''']);
    end;
  end;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Actual implementation

N = size(data, 1);
F = size(data, 2);

if N > F
    basis = pcomp_std(data, scale);
else
    basis = pcomp_trick(data, scale);
end

end

