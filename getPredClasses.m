function [predMat, predLbls, trueTruePos, falseTruePos, acc, confMat, order] = ...
    getPredClasses( classifier, hists, testLbls, trueLblNums, ...
    falseLblNums, predMat )
%GETCLASSIFICATIONS Summary of this function goes here
%   Detailed explanation goes here

nTestImgs = size(predMat, 1);

% Filter labels
combinedLblNums = [trueLblNums falseLblNums];
filtLbls = false(nTestImgs, 1);
for i = 1:nTestImgs
    for L = combinedLblNums
        if predMat(i, L)
            filtLbls(i) = true;
            break
        end
    end
end
nTestFiltImgs = sum(filtLbls);

% Classify
predLbls = svmclassify(classifier, hists(filtLbls, :));

% Calculate classification accuracy
trueTestLbls = ismember(testLbls(filtLbls), trueLblNums);
falseTestLbls = ismember(testLbls(filtLbls), falseLblNums);
trueTruePos  = sum( predLbls & trueTestLbls);
falseTruePos = sum(~predLbls & falseTestLbls);
acc = (trueTruePos + falseTruePos) / nTestFiltImgs;
fprintf('Accuracy: %.02f%%\n', acc * 100)

% Create confusion matrix
[confMat, order] = confusionmat(trueTestLbls, predLbls);
% maxCountTruePos = max(sum(testLbls), sum(~testLbls_IP));
% figure, imagesc(confMat_IPvACRS, [1 maxCountTruePos]), colorbar

% Update prediction matrix
i2 = 1;
for i1 = 1:nTestImgs
    if filtLbls(i1)
        for L = trueLblNums
            predMat(i1, L) = predLbls(i2);
        end
        for L = falseLblNums
            predMat(i1, L) = ~predLbls(i2);
        end
        i2 = i2 + 1;
    end
end
