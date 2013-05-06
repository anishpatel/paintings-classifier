% Painting Classification
%   Terry Rabinowitz
%   Anish Patel

clear all
run_vl_setup
setParams

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
% trainHists = ...
%     getConcatHists(trainSiftHists, trainGlcmFeats, trainHsvHists, trainRgbHists, trainLineFeats);
% testHists = ...
%     getConcatHists(testSiftHists, testGlcmFeats, testHsvHists, testRgbHists, testLineFeats);
trainHists = trainLineFeats;
testHists = testLineFeats;

%% Train Classifiers - create a binary classifier for each pair of labels
trainLabelPairs = cell(nLbls, nLbls);
svmClassifiers = cell(nLbls, nLbls);
for L1 = labels
    for L2 = L1+1:nLbls
        L1_lbls = trainLbls == L1;
        L2_lbls = trainLbls == L2;
        trainLabelPairs{L1,L2} = [trainLbls(L1_lbls); trainLbls(L2_lbls)];
        trainHistsPair = [trainHists(L1_lbls,:); trainHists(L2_lbls,:)];
        svmClassifiers{L1,L2} = svmtrain(trainHistsPair, trainLabelPairs{L1,L2});
    end
end

%% Classify Test - classify test images
predLabelPairs = cell(nLbls, nLbls);
predLblHists = zeros(nTestImgs, nLbls); 
for L1 = labels
    for L2 = L1+1:nLbls
        predLabelPairs{L1,L2} = svmclassify(svmClassifiers{L1,L2}, testHists);
        for i = 1:nTestImgs
            predLbl = predLabelPairs{L1,L2}(i);
            predLblHists(i,predLbl) = predLblHists(i,predLbl) + 1;
        end
    end
end

%% Results - evaluate results with accuracy measure and confusion matrix
% load('data/results.mat')
[~, predLbls] = arrayfun(@(i) max(predLblHists(i,:)), (1:nTestImgs)');
[confMat, order] = confusionmat(testLbls, predLbls);
accuracy = sum(predLbls == testLbls) / nTestImgs;
save('data/results.mat', 'predLbls', 'confMat', 'order', 'accuracy')

figure, imagesc(confMat, [1 nTestImgsPerLbl]), colorbar
fprintf('Accuracy: %.02f%%\n', accuracy*100)
