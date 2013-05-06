function [ glcmFeats ] = getGlcmFeats( images )
%GETGLCMHISTS Summary of this function goes here
%   Detailed explanation goes here

nImgs = length(images);
offsetRange = 10;
numLevels = 48;
glcmFeats = zeros(nImgs, 2*offsetRange*4);
for i = 1:nImgs
    img = images{i};
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
   
    offsets0 = [zeros(offsetRange,1) (1:offsetRange)'];
    offsets1 = [(1:offsetRange)' zeros(offsetRange,1)];
    glcms = graycomatrix(img,'Offset',offsets0,'NumLevels',numLevels);
    glcmsVert = graycomatrix(img,'Offset',offsets1,'NumLevels',numLevels);
    %stats = GLCM_Features1(glcms);
    stats = graycoprops(glcms);
    %statsVert = GLCM_Features1(glcmsVert);
    statsVert = graycoprops(glcmsVert);
    temp = cell2mat(struct2cell(stats));
    tempVert = cell2mat(struct2cell(statsVert));
    glcmFeats(i,:) = [temp(:)' tempVert(:)'];
end
