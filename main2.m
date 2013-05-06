% Painting Classification
%   Terry Rabinowitz
%   Anish Patel

clear all
run_vl_setup
setParams
tic

%% Read Images - read images from files and pre-process
load('data/images.mat')
% fprintf('Reading images\n')
% [trainImgs, trainLbls, testImgs, testLbls] = getImgs(gaussFilt); % smoothes each image after read
% trainSmlImgs = getSmallImgs(trainImgs);
% testSmlImgs = getSmallImgs(testImgs);
% save('data/images.mat', 'trainImgs', 'testImgs', 'trainLbls', 'testLbls', 'trainSmlImgs', 'testSmlImgs')

%% Train SIFT - generate SIFT features for each training image
% load('data/trainSiftFeats.mat')
% fprintf('Generating training SIFT features\n')
% [trainSiftDescrs, nTrainSiftFeatsPerImg, trainSiftDescrsPerImg] = getSiftFeats(trainSmlImgs);
% save('data/trainSiftFeats.mat', 'trainSiftDescrs', 'nTrainSiftFeatsPerImg', 'trainSiftDescrsPerImg')

% Train SIFT Vocab - build a SIFT feature vocabulary and create a SIFT words histogram for each training image
load('data/trainSiftHists.mat')
% fprintf('Building sift vocabulary and creating train sift word histograms\n')
% [siftVocab, trainSiftHists] = getVocab(nTrainImgs, trainSiftDescrs, nTrainSiftFeatsPerImg);
% save('data/trainSiftHists.mat', 'siftVocab', 'trainSiftHists')

%% Test SIFT - generate SIFT features for each testing image
% load('data/testSiftFeats.mat')
% fprintf('creating test sift features\n')
% [testSiftDescrs, nTestSiftFeatsPerImg, testSiftDescrsPerImg] = getSiftFeats(testSmlImgs);
% save('data/testSiftFeats.mat', 'testSiftDescrs', 'nTestSiftFeatsPerImg', 'testSiftDescrsPerImg')

% Test SIFT Hists - create a SIFT words histogram for each testing image
% TODO use histc
% TODO use kd-trees
load('data/testSiftHists.mat')
% fprintf('creating test sift word histograms\n')
% testSiftHists = getSiftWordHists(nTestImgs, siftVocab, testSiftDescrsPerImg, nTestSiftFeatsPerImg);
% save('data/testSiftHists.mat', 'testSiftHists')

%% GLCM Feats
load('data/GlcmFeats.mat')
% trainGlcmFeats = getGlcmFeats(trainImgs);
% testGlcmFeats = getGlcmFeats(testImgs);
% save('data/GlcmFeats.mat', 'trainGlcmFeats', 'testGlcmFeats')

%% HSV Hists - calculate HSV histograms for each image
load('data/HsvHists.mat')
% trainHsvHists = getHsvHists(trainImgs);
% testHsvHists = getHsvHists(testImgs);
% save('data/HsvHists.mat', 'trainHsvHists', 'testHsvHists')

%% RGB Hists - calculate RGB histograms for each image
load('data/RgbHists.mat')
% trainRgbHists = getRgbHists(trainImgs);
% testRgbHists = getRgbHists(testImgs);
% save('data/RgbHists.mat', 'trainRgbHists', 'testRgbHists')

%% Line Feats
load('data/LineFeats.mat')
% trainLineFeats = getLineFeats(trainImgs);
% testLineFeats = getLineFeats(testImgs);
% save('data/LineFeats.mat', 'trainLineFeats', 'testLineFeats')

%% Edge Hists - calculate egde gradient histograms for each image
% load('data/EdgeHists.mat')
% trainEdgeHists = getEdgeGradHists(trainImgs);
% testEdgeHists = getEdgeGradHists(testImgs);
% save('data/EdgeHists.mat', 'trainEdgeHists', 'testEdgeHists')

%% Concat Hists - concat various histograms
[trainHists, siftCols, glcmCols, hsvCols, rgbCols, lineCols] = ...
    getConcatHists(trainSiftHists, trainGlcmFeats, trainHsvHists, trainRgbHists, trainLineFeats);
testHists = ...
    getConcatHists(testSiftHists, testGlcmFeats, testHsvHists, testRgbHists, testLineFeats);

predMat = ones(nLbls, nTestImgs);
truePos = zeros(nLbls, 1);

%% Impressionism Pointillism vs. Abstract Cubism Renaissance Surrealism
trueLblNums = [I P];
falseLblNums = [A C R S];
histCols_IPvACRS = siftCols|glcmCols|rgbCols|lineCols;

classifier_IPvACRS = getClassifier( ...
    trainHists(:, histCols_IPvACRS), trainLbls, trueLblNums, falseLblNums );

fprintf('%7s ', 'IPvACRS')
[predMat, predLbls_IPvACRS, ~, ~, acc_IPvACRS, confMat_IPvACRS, ~] = ...
    getPredClasses( classifier_IPvACRS, testHists(:, histCols_IPvACRS), ...
    testLbls, trueLblNums, falseLblNums, predMat );

%% Abstract Cubism vs. Renaissance Surrealism
histCols_ACvRS = siftCols|glcmCols|hsvCols;

trainLbls_ACvRS = trainLbls(~trainLbls_IPvACRS);
trainLbls_ACvRS = trainLbls_ACvRS == A | trainLbls_ACvRS == C;
trainHists_ACvRS = trainHists(~trainLbls_IPvACRS, histCols_ACvRS);
classifier_ACvRS = svmtrain(trainHists_ACvRS, trainLbls_ACvRS);

testHists_ACvRS = testHists(~predLbls_IPvACRS, histCols_ACvRS);
predLbls_ACvRS = svmclassify(classifier_ACvRS, testHists_ACvRS);

nTestImgs_ACvsRS = sum(~predLbls_IPvACRS);
testLbls_AC = testLbls(~predLbls_IPvACRS);
testLbls_AC = testLbls_AC == A | testLbls_AC == C;
acc_ACvRS = sum(predLbls_ACvRS == testLbls_AC) / nTestImgs_ACvsRS;
[confMat_ACvRS, ~] = confusionmat(testLbls_AC, predLbls_ACvRS);

fprintf('Accuracy ACvRS: %.02f%%\n', acc_ACvRS*100)
maxCount_ACvRS = max(sum(testLbls_AC), sum(~testLbls_AC));
% figure, imagesc(confMat_ACvRS, [1 nTestImgs_ACvsRS]), colorbar

%% Impressionism vs. Pointillism
histCols_IvP = siftCols|rgbCols;

trainLbls_IvP = trainLbls(trainLbls_IPvACRS) == I;
trainHists_IvP = trainHists(trainLbls_IPvACRS, histCols_IvP);
classifier_IvP = svmtrain(trainHists_IvP, trainLbls_IvP);

testHists_IvP = testHists(predLbls_IPvACRS, histCols_IvP);
predLbls_IvP = svmclassify(classifier_IvP, testHists_IvP);

nTestImgs_IvP = sum(predLbls_IPvACRS);
testLbls_I = testLbls(predLbls_IPvACRS) == I;
testLbls_P = testLbls(predLbls_IPvACRS) == P;
acc_IvP = sum(predLbls_IvP == testLbls_I) / nTestImgs_IvP;
[confMat_IvP, ~] = confusionmat(testLbls_I, predLbls_IvP);

fprintf('Accuracy IvP: %.02f%%\n', acc_IvP*100)
maxCount_IvP = max(sum(testLbls_I), sum(~testLbls_I));
% figure, imagesc(confMat_IvP, [1 maxCount_IvP]), colorbar

truePos(I) = sum(predLbls_IvP & testLbls_I);
truePos(P) = sum(~predLbls_IvP & ~testLbls_P);

%% Abstract vs. Cubism
histCols_AvC = siftCols|glcmCols|rgbCols|lineCols;

trainLbls_AvC = trainLbls(~trainLbls_IPvACRS);
trainLbls_AvC = trainLbls_AvC(trainLbls_ACvRS) == A;
trainHists_AvC = trainHists(~trainLbls_IPvACRS, histCols_AvC);
trainHists_AvC = trainHists_AvC(trainLbls_ACvRS, :);
classifier_AvC = svmtrain(trainHists_AvC, trainLbls_AvC);

testHists_AvC = testHists(~predLbls_IPvACRS, histCols_AvC);
testHists_AvC = testHists_AvC(predLbls_ACvRS, :);
predLbls_AvC = svmclassify(classifier_AvC, testHists_AvC);

nTestImgs_AvC = sum(predLbls_ACvRS);
testLbls_A = testLbls(~predLbls_IPvACRS);
testLbls_A = testLbls_A(predLbls_ACvRS) == A;
testLbls_C = testLbls(~predLbls_IPvACRS);
testLbls_C = testLbls_C(predLbls_ACvRS) == C;
acc_AvC = sum(predLbls_AvC == testLbls_A) / nTestImgs_AvC;
[confMat_AvC, ~] = confusionmat(testLbls_A, predLbls_AvC);

fprintf('Accuracy AvC: %.02f%%\n', acc_AvC*100)
maxCount_AvC = max(sum(testLbls_A), sum(~testLbls_A));
% figure, imagesc(confMat_AvC, [1 maxCount_AvC]), colorbar

truePos(A) = sum(predLbls_AvC & testLbls_A);
truePos(C) = sum(~predLbls_AvC & ~testLbls_C);

%% Renaissance vs. Surrealism
histCols_RvS = siftCols|hsvCols|lineCols;

trainLbls_RvS = trainLbls(~trainLbls_IPvACRS);
trainLbls_RvS = trainLbls_RvS(~trainLbls_ACvRS) == R;
trainHists_RvS = trainHists(~trainLbls_IPvACRS, histCols_RvS);
trainHists_RvS = trainHists_RvS(~trainLbls_ACvRS, :);
classifier_RvS = svmtrain(trainHists_RvS, trainLbls_RvS);

testHists_RvS = testHists(~predLbls_IPvACRS, histCols_RvS);
testHists_RvS = testHists_RvS(~predLbls_ACvRS, :);
predLbls_RvS = svmclassify(classifier_RvS, testHists_RvS);

nTestImgs_RvS = sum(~predLbls_ACvRS);
testLbls_R = testLbls(~predLbls_IPvACRS);
testLbls_R = testLbls_R(~predLbls_ACvRS) == R;
testLbls_S = testLbls(~predLbls_IPvACRS);
testLbls_S = testLbls_S(~predLbls_ACvRS) == S;
acc_RvS = sum(predLbls_RvS == testLbls_R) / nTestImgs_RvS;
[confMat_RvS, ~] = confusionmat(testLbls_R, predLbls_RvS);

fprintf('Accuracy RvS: %.02f%%\n', acc_RvS*100)
maxCount_RvS = max(sum(testLbls_R), sum(~testLbls_R));
% figure, imagesc(confMat_RvS, [1 maxCount_RvS]), colorbar

truePos(R) = sum(predLbls_RvS & testLbls_R);
truePos(S) = sum(~predLbls_RvS & ~testLbls_S);

%% True Positive Results
fprintf('True Positives:\n')
for L = labels
    fprintf('%13s  %d\n', labelNames{L}, truePos(L))
end
totalAcc = sum(truePos) / nTestImgs;
fprintf('Total Accuracy: %.02f%%\n', totalAcc*100)
