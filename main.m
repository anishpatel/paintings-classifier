% Painting Classification
%   Terry Rabinowitz
%   Anish Patel

clear all
run_vl_setup
setParams

%% Read Images - read images from files and pre-process
% load('data/images.mat')
fprintf('Reading images\n')
[trainImgs, trainLbls, testImgs, testLbls] = getImgs(gaussFilt); % smoothes each image after read
trainSmlImgs = getSmallImgs(trainImgs);
testSmlImgs = getSmallImgs(testImgs);
save('data/images.mat', 'trainImgs', 'testImgs', 'trainLbls', ...
    'testLbls', 'trainSmlImgs', 'testSmlImgs')

%% Train SIFT - generate training SIFT features and build vocabulary
% load('data/trainSiftFeats.mat')
fprintf('Generating training SIFT features\n')
[trainSiftDescrs, nTrainSiftFeatsPerImg, trainSiftDescrsPerImg] = getSiftFeats(trainSmlImgs);
save('data/trainSiftFeats.mat', 'trainSiftDescrs', ...
    'nTrainSiftFeatsPerImg', 'trainSiftDescrsPerImg')

% Train SIFT Vocab 
% build a SIFT feature vocabulary and create a SIFT words histogram for each training image
% load('data/trainSiftHists.mat')
fprintf('Building sift vocabulary and creating train sift word histograms\n')
[siftVocab, trainSiftHists] = getVocab(nTrainImgs, trainSiftDescrs, nTrainSiftFeatsPerImg);
save('data/trainSiftHists.mat', 'siftVocab', 'trainSiftHists')

%% Test SIFT - generate SIFT features for each testing image
% load('data/testSiftFeats.mat')
fprintf('creating test sift features\n')
[testSiftDescrs, nTestSiftFeatsPerImg, testSiftDescrsPerImg] = getSiftFeats(testSmlImgs);
save('data/testSiftFeats.mat', 'testSiftDescrs', 'nTestSiftFeatsPerImg', 'testSiftDescrsPerImg')

% Test SIFT Hists - create a SIFT words histogram for each testing image
% TODO use histc
% TODO use kd-trees
% load('data/testSiftHists.mat')
fprintf('creating test sift word histograms\n')
testSiftHists = getSiftWordHists( nTestImgs, siftVocab, ...
    testSiftDescrsPerImg, nTestSiftFeatsPerImg);
save('data/testSiftHists.mat', 'testSiftHists')

%% GLCM Feats
% load('data/GlcmFeats.mat')
trainGlcmFeats = getGlcmFeats(trainImgs);
testGlcmFeats = getGlcmFeats(testImgs);
save('data/GlcmFeats.mat', 'trainGlcmFeats', 'testGlcmFeats')

%% RGB Hists - calculate RGB histograms for each image
% load('data/RgbHists.mat')
trainRgbHists = getRgbHists(trainImgs);
testRgbHists = getRgbHists(testImgs);
save('data/RgbHists.mat', 'trainRgbHists', 'testRgbHists')

%% HSV Hists - calculate HSV histograms for each image
% load('data/HsvHists.mat')
trainHsvHists = getHsvHists(trainImgs);
testHsvHists = getHsvHists(testImgs);
save('data/HsvHists.mat', 'trainHsvHists', 'testHsvHists')

%% Line Feats - calculates number of lines and avg. line length
% load('data/LineFeats.mat')
trainLineFeats = getLineFeats(trainImgs);
testLineFeats = getLineFeats(testImgs);
save('data/LineFeats.mat', 'trainLineFeats', 'testLineFeats')

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

predMat = true(nTestImgs, nLbls);
truePos = zeros(nLbls, 1);

%% Abstract Cubism vs. Impressionism Pointillism Renaissance Surrealism
trueLblNums  = [A C];
falseLblNums = [I P R S];
histCols_ACvIPRS = siftCols|glcmCols|rgbCols|hsvCols|lineCols;

classifier_ACvIPRS = getClassifier( ...
    trainHists(:, histCols_ACvIPRS), trainLbls, trueLblNums, falseLblNums );

fprintf('%7s ', 'ACvIPRS')
[predMat, predLbls_ACvIPRS, ~, ~, acc_ACvIPRS, confMat_ACvIPRS, ~] = ...
    getPredClasses( classifier_ACvIPRS, testHists(:, histCols_ACvIPRS), ...
    testLbls, trueLblNums, falseLblNums, predMat );

%% Impressionism Pointillism vs. Renaissance Surrealism
trueLblNums  = [I P];
falseLblNums = [R S];
histCols_IPvRS = siftCols|glcmCols|rgbCols|hsvCols|lineCols;

classifier_IPvRS = getClassifier( ...
    trainHists(:, histCols_IPvRS), trainLbls, trueLblNums, falseLblNums );

fprintf('%7s ', 'IPvRS')
[predMat, predLbls_IPvRS, ~, ~, acc_IPvRS, confMat_IPvRS, ~] = ...
    getPredClasses( classifier_IPvRS, testHists(:, histCols_IPvRS), ...
    testLbls, trueLblNums, falseLblNums, predMat );

%% Abstract vs. Cubism
trueLblNums  = A;
falseLblNums = C;
histCols_AvC = siftCols|glcmCols|rgbCols|hsvCols|lineCols;

classifier_AvC = getClassifier( ...
    trainHists(:, histCols_AvC), trainLbls, trueLblNums, falseLblNums );

fprintf('%7s ', 'AvC')
[predMat, predLbls_AvC, truePos(trueLblNums), truePos(falseLblNums), acc_AvC, confMat_AvC, ~] = ...
    getPredClasses( classifier_AvC, testHists(:, histCols_AvC), ...
    testLbls, trueLblNums, falseLblNums, predMat );

%% Impressionism vs. Pointillism
trueLblNums  = I;
falseLblNums = P;
histCols_IvP = siftCols|glcmCols|rgbCols|hsvCols|lineCols;

classifier_IvP = getClassifier( ...
    trainHists(:, histCols_IvP), trainLbls, trueLblNums, falseLblNums );

fprintf('%7s ', 'IvP')
[predMat, predLbls_IvP, truePos(trueLblNums), truePos(falseLblNums), acc_IvP, confMat_IvP, ~] = ...
    getPredClasses( classifier_IvP, testHists(:, histCols_IvP), ...
    testLbls, trueLblNums, falseLblNums, predMat );

%% Renaissance vs. Surrealism
trueLblNums  = R;
falseLblNums = S;
histCols_RvS = siftCols|glcmCols|rgbCols|hsvCols|lineCols;

classifier_RvS = getClassifier( ...
    trainHists(:, histCols_RvS), trainLbls, trueLblNums, falseLblNums );

fprintf('%7s ', 'RvS')
[predMat, predLbls_RvS, truePos(trueLblNums), truePos(falseLblNums), acc_RvS, confMat_RvS, ~] = ...
    getPredClasses( classifier_RvS, testHists(:, histCols_RvS), ...
    testLbls, trueLblNums, falseLblNums, predMat );

%% True Positive Results
fprintf('True Positives:\n')
for L = labels
    fprintf('%13s  %d\n', labelNames{L}, truePos(L))
end
totalAcc = sum(truePos) / nTestImgs;
fprintf('Total Accuracy: %.02f%%\n', totalAcc*100)

predLbls = zeros(nTestImgs, 1);
for i = 1:nTestImgs
    for L = 1:nLbls
        if predMat(i, L)
            predLbls(i) = L;
            break
        end
    end
end

[confMat, order] = confusionmat(testLbls, predLbls);
figure, imagesc(confMat, [1 50]), colorbar


