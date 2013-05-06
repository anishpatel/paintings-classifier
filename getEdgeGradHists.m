function [ edgeHists ] = getEdgeGradHists( images )
%GETEDGEGRADHISTS Summary of this function goes here
%   Detailed explanation goes here

global nHistBins edgeFilt
nImgs = length(images);

edgeHists = zeros(nImgs, nHistBins);
for i = 1:nImgs
    img = images{i};
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    edgeGradImg = imfilter(img, edgeFilt);
    edgeGradHist = imhist(edgeGradImg, nHistBins+1);
    edgeHists(i,:) = edgeGradHist(2:end);
end
