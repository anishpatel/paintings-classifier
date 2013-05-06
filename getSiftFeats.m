function [allDescrs, nFeatsPerImg, descrsPerImg] = getSiftFeats(images)
%SIFTFEATS Generate SIFT descriptors for each image
%   Detailed explanation goes here

global nDsiftSteps
nImgs = length(images);

% Calculate dense SIFT feature descriptors for each image
descrsPerImg = cell(nImgs, 1);
nFeatsPerImg = cell(nImgs, 1);
nFeats = 0;
for i = 1:nImgs
    img = images{i};
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    [~, descrsPerImg{i}] = vl_dsift(single(img), 'step', nDsiftSteps, 'fast');
    nFeatsPerImg{i} = size(descrsPerImg{i}, 2);
    nFeats = nFeats + nFeatsPerImg{i};
end

% Cat all features into a single matrix
allDescrs = zeros(128, nFeats, 'double');
f = 1;
for i = 1:nImgs
    allDescrs(:,f:f-1+nFeatsPerImg{i}) = descrsPerImg{i};
    f = f + nFeatsPerImg{i};
end
