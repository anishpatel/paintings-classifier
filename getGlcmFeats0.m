function [ glcmFeats ] = getGlcmFeats( images )
%GETGLCMHISTS Summary of this function goes here
%   Detailed explanation goes here

nImgs = length(images);

glcmFeats = zeros(nImgs, 22);
for i = 1:nImgs
    img = images{i};
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    glcm = graycomatrix(img);
    stats = GLCM_Features1(glcm);
    glcmFeats(i,:) = cell2mat(struct2cell(stats));
end
