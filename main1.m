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

%% Train SIFT Vocab - build a SIFT feature vocabulary and create a SIFT words histogram for each training image
load('data/trainSiftHists.mat')
% fprintf('Building sift vocabulary and creating train sift word histograms\n')
% [siftVocab, trainSiftHists] = getVocab(nTrainImgs, trainSiftDescrs, nTrainSiftFeatsPerImg);
% save('data/trainSiftHists.mat', 'siftVocab', 'trainSiftHists')

%% Test SIFT - generate SIFT features for each testing image
% load('data/testSiftFeats.mat')
% fprintf('creating test sift features\n')
% [testSiftDescrs, nTestSiftFeatsPerImg, testSiftDescrsPerImg] = getSiftFeats(testSmlImgs);
% save('data/testSiftFeats.mat', 'testSiftDescrs', 'nTestSiftFeatsPerImg', 'testSiftDescrsPerImg')

%% Test SIFT Hists - create a SIFT words histogram for each testing image
% TODO use histc
load('data/testSiftHists.mat')
% fprintf('creating test sift word histograms\n')
% testSiftHists = getSiftWordHists(nTestImgs, siftVocab, testSiftDescrsPerImg, nTestSiftFeatsPerImg);
% save('data/testSiftHists.mat', 'testSiftHists')

%% GLCM Feats
load('data/GlcmFeats.mat')
% trainGlcmFeats = getGlcmFeats(trainImgs);
% testGlcmFeats = getGlcmFeats(testImgs);
% save('data/GlcmFeats.mat', 'trainGlcmFeats', 'testGlcmFeats')

%% Color Hists - calculate color histograms for each image
% load('data/ColorHists.mat')
trainColorHists = getColorHists(trainImgs);
testColorHists = getColorHists(testImgs);
save('data/ColorHists.mat', 'trainColorHists', 'testColorHists')

%% Train Line Feats
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
% clear trainImgs trainSmlImgs testImgs testSmlImgs
trainHists = [ ...
    trainSiftHists ...
    trainGlcmFeats ...
    trainColorHists ...
    trainLineFeats ...
%     trainEdgeHists ...
    ];
testHists = [ ...
    testSiftHists ...
    testGlcmFeats ...
    testColorHists ...
    testLineFeats ...
%     testEdgeHists ...
    ];

%% Impressionism Pointillism vs. Abstract Cubism Renaissance Surrealism
trainLblsPI = (trainLbls == 3) | (trainLbls == 4); % impressionism and pointillism
trainHistsPI = trainHists;
classifierPI = svmtrain(trainHistsPI, trainLblsPI);

testHistsPI = testHists;
predLblsPI = svmclassify(classifierPI, testHistsPI);

nLeftoverImgsPI = nTestImgs;
testLblsPI = trainLblsPI;
[confMatPI, ~] = confusionmat(testLblsPI, predLblsPI);
accuracyPI = sum(predLblsPI == testLblsPI) / nLeftoverImgsPI;
figure, imagesc(confMatPI, [1 200]), colorbar
fprintf('Accuracy: %.02f%%\n', accuracyPI*100)

%% Abstract Cubism vs. Renaissance Surrealism
trainLblsAC = ((trainLbls == 1) | (trainLbls == 2)); % Abstract Cubism
trainLblsAC = trainLblsAC(~trainLblsPI);
trainHistsAC = trainHistsPI(~trainLblsPI,:);
classifierAC = svmtrain(trainHistsAC, trainLblsAC);

testHistsAC = testHistsPI(~predLblsPI,:);
predLblsAC = svmclassify(classifierAC, testHistsAC);

nLeftoverImgsAC = sum(~predLblsPI);
testLblsAC = testLbls(~predLblsPI);
for i = 1:nLeftoverImgsAC
    if (testLblsAC(i) == 1 || testLblsAC(i) == 2)
        testLblsAC(i) = 1;
    else
        testLblsAC(i) = 0;
    end
end
testLblsAC = logical(testLblsAC);
[confMatAC, ~] = confusionmat(testLblsAC, predLblsAC);
accuracyAC = sum(predLblsAC == testLblsAC) / nLeftoverImgsAC;
figure, imagesc(confMatAC, [1 100]), colorbar
fprintf('Accuracy: %.02f%%\n', accuracyAC*100)

%% Impressionism vs. Pointillism
trainLblsI = trainLbls == 3; % Impressionism
trainLblsI = trainLblsI(trainLblsPI);
trainHistsI = trainHistsPI(trainLblsPI,:);
classifierI = svmtrain(trainHistsI, trainLblsI);

testHistsI = testHistsPI(predLblsPI,:);
predLblsI = svmclassify(classifierI, testHistsI);

nLeftoverImgsI = sum(predLblsPI);
testLblsI = testLbls == 3;
testLblsI = testLblsI(predLblsPI);
testLblsP = testLbls == 4;
testLblsP = testLblsP(predLblsPI);
[confMatI, ~] = confusionmat(testLblsI, predLblsI);
truePosI = sum(predLblsI & testLblsI);
truePosP = sum(~predLblsI & ~testLblsP);
fprintf('True Positives I: %d\n', truePosI)
fprintf('True Positives P: %d\n', truePosP)
accuracyI = sum(predLblsI == testLblsI) / nLeftoverImgsI;
figure, imagesc(confMatI, [1 50]), colorbar
fprintf('Accuracy: %.02f%%\n', accuracyI*100)

%% Abstract vs. Cubism
trainLblsA = trainLbls == 1; % Abstract
trainLblsA = trainLblsA(~trainLblsPI);
trainLblsA = trainLblsA(trainLblsAC);
trainHistsA = trainHistsAC(trainLblsAC,:);
classifierA = svmtrain(trainHistsA, trainLblsA);

testHistsA = testHistsAC(predLblsAC,:);
predLblsA = svmclassify(classifierA, testHistsA);

nLeftoverImgsA = sum(predLblsAC);
testLblsA = testLbls == 1;
testLblsA = testLblsA(~predLblsPI);
testLblsA = testLblsA(predLblsAC);
testLblsC = testLbls == 2;
testLblsC = testLblsC(~predLblsPI);
testLblsC = testLblsC(predLblsAC);
[confMatA, ~] = confusionmat(testLblsA, predLblsA);
truePosA = sum(predLblsA & testLblsA);
truePosC = sum(~predLblsA & ~testLblsC);
fprintf('True Positives A: %d\n', truePosA)
fprintf('True Positives C: %d\n', truePosC)
accuracyA = sum(predLblsA == testLblsA) / nLeftoverImgsA;
figure, imagesc(confMatA, [1 50]), colorbar
fprintf('Accuracy: %.02f%%\n', accuracyA*100)

%% Renaissance vs. Surrealism
trainLblsR = trainLbls == 5; % Renaissance
trainLblsR = trainLblsR(~trainLblsPI);
trainLblsR = trainLblsR(~trainLblsAC);
trainHistsR = trainHistsAC(~trainLblsAC,:);
classifierR = svmtrain(trainHistsR, trainLblsR);

testHistsR = testHistsAC(~predLblsAC,:);
predLblsR = svmclassify(classifierR, testHistsR);

nLeftoverImgsR = sum(~predLblsAC);
testLblsR = testLbls == 5;
testLblsR = testLblsR(~predLblsPI);
testLblsR = testLblsR(~predLblsAC);
testLblsS = testLbls == 6;
testLblsS = testLblsS(~predLblsPI);
testLblsS = testLblsS(~predLblsAC);
[confMatR, ~] = confusionmat(testLblsR, predLblsR);
truePosR = sum(predLblsR & testLblsR);
truePosS = sum(~predLblsR & ~testLblsS);
fprintf('True Positives R: %d\n', truePosR)
fprintf('True Positives S: %d\n', truePosS)
accuracyR = sum(predLblsR == testLblsR) / nLeftoverImgsR;
figure, imagesc(confMatR, [1 50]), colorbar
fprintf('Accuracy: %.02f%%\n', accuracyR*100)

%% Train and Classify - Abstract Cubism Renaissance Surrealism
% trainLabelPairs = cell(nLbls, nLbls);
% svmClassifiers = cell(nLbls, nLbls);
% restLbls = [1 5 6];
% for L1 = restLbls
%     for L2 = restLbls
%         if L1 == L2
%             break
%         end
%         L1_lbls = trainLbls == L1;
%         L2_lbls = trainLbls == L2;
%         trainLabelPairs{L1,L2} = [trainLbls(L1_lbls); trainLbls(L2_lbls)];
%         trainHistsPair = [trainHists(L1_lbls,:); trainHists(L2_lbls,:)];
%         svmClassifiers{L1,L2} = svmtrain(trainHistsPair, trainLabelPairs{L1,L2});
%     end
% end
% 
% predLabelPairs = cell(nLbls, nLbls);
% predLblHists = zeros(nTestImgs, nLbls); 
% for L1 = restLbls
%     for L2 = restLbls
%         if L1 == L2
%             break
%         end
%         predLabelPairs{L1,L2} = svmclassify(svmClassifiers{L1,L2}, testHists);
%         for i = 1:nTestImgs
%             predLbl = predLabelPairs{L1,L2}(i);
%             predLblHists(i,predLbl) = predLblHists(i,predLbl) + 1;
%         end
%     end
% end
% 
% [~, predLbls] = arrayfun(@(i) max(predLblHists(i,:)), (1:nTestImgs)');
% [confMat, ~] = confusionmat(testLbls, predLbls);
% accuracy = sum(predLbls == testLbls) / nTestImgs;
% figure, imagesc(confMat, [1 nTestImgsPerLbl]), colorbar
% fprintf('Accuracy: %.02f%%\n', accuracy*100)
