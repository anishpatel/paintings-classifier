function [rgbHists] = getRgbHists(images)
%GETRGBHISTS Calculates a color histogram for each image
%   Note: The color histogram is a concatenation of the histograms of each 
%   of the RGB color channels.

global nHistBins
nImgs = length(images);

rgbHists = zeros(nImgs, 3 * nHistBins);
for i = 1:nImgs
    img = images{i};
    if size(img, 3) == 3
        redHist = imhist(img(:,:,1), nHistBins);
        grnHist = imhist(img(:,:,2), nHistBins);
        bluHist = imhist(img(:,:,3), nHistBins);
        rgbHists(i, :) = [redHist; grnHist; bluHist]';
    else
        grayHist = imhist(img, nHistBins);
        rgbHists(i, :) = [grayHist; grayHist; grayHist]';
    end
end
