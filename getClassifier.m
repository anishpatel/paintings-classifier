function [classifier] = getClassifier(hists, trainLbls, trueLblNums, falseLblNums)
%GETCLASSIFIER Summary of this function goes here
%   Detailed explanation goes here

% Filter labels
combinedLblNums = [trueLblNums falseLblNums];
filtLbls = ismember(trainLbls, combinedLblNums);
trueLbls = ismember(trainLbls(filtLbls), trueLblNums);

% Create classifier
classifier = svmtrain(hists(filtLbls, :), trueLbls);
