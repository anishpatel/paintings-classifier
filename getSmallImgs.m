function [smallImgs] = getSmallImgs(images)
%GETSMALLIMGS Summary of this function goes here
%   Detailed explanation goes here

global smallSize
nImgs = length(images);

smallImgs = cell(nImgs, 1);
for i = 1:nImgs
    smallImgs{i} = imresize(images{i}, smallSize);
end
