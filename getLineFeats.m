function [ lineFeats ] = getLineFeats( images )
%GETLINES Summary of this function goes here
%   Detailed explanation goes here

nImgs = length(images);

lineFeats = zeros(nImgs, 2);
for i = 1:nImgs
    img = images{i};
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    BW = edge(img,'canny');
    [H,T,R] = hough(BW);
    P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
    % Find lines and plot them
    lines = houghlines(BW,T,R,P,'FillGap',5,'MinLength',7);
    
    % Create features
    nLines = length(lines);
    distSum = 0;
    for k = 1:nLines
        distSum = distSum + norm(lines(k).point1 - lines(k).point2); % Euclidean distance
    end
    imgDiag = norm(size(img));
    avgLength = distSum / nLines;
    avgLength = avgLength / imgDiag; % normalize by image size
    lineFeats(i,:) = [nLines avgLength];
end
