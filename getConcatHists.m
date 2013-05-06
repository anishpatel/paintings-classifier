function [ hists, varargout ] = getConcatHists( varargin )
%GETCONCATHISTS Summary of this function goes here
%   Detailed explanation goes here

% Concat all input
hists = [varargin{:}];

% Give columns of the section of each input in concatted matrix
if nargout > 1
    histCumSizes = cumsum(cellfun(@(X) size(X,2), varargin));
    histSize = histCumSizes(end);
    varargout = cell(nargout-1, 1);
    prevSize = 0;
    for k = 1:nargout-1
        varargout{k} = false(1, histSize);
        varargout{k}(prevSize+1:histCumSizes(k)) = true;
        prevSize = histCumSizes(k);
    end
end
