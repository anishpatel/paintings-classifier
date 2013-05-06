function [hsvHists] = getHsvHists(images)
%GETHSVHISTS Calculates a hsv histogram for each image
%   Note: The hsv histogram is a concatenation of the histograms of each 
%   of the HSV channels.

global nHistBins
nImgs = length(images);

hsvHists = zeros(nImgs, 3 * nHistBins);
for i = 1:nImgs
    img = images{i};
    if size(img, 3) == 3
        % color image
        hsvImg = rgb2hsv(img);
        hueHist = imhist(hsvImg(:,:,1), nHistBins);
        satHist = imhist(hsvImg(:,:,2), nHistBins);
        valHist = imhist(hsvImg(:,:,3), nHistBins);
        hsvHists(i, :) = [hueHist; satHist; valHist]';
    else
        % gray image
        grayHist = imhist(img, nHistBins);
        hsvHists(i, :) = [zeros(nHistBins,1); zeros(nHistBins,1); grayHist]';
    end
end
