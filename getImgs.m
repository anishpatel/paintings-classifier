function [trainImgs, trainLbls, testImgs, testLbls] = getImgs(gaussFilt)
%GETIMAGES Reads and smoothes images from file
%   Detailed explanation goes here

global nTrainImgs nTestImgs nLbls nArtistsPerLbl nTrainImgsPerArtist 
global nTestImgsPerArtist paintPath labelNames imgNameExt
nImgsPerArtist = nTrainImgsPerArtist + nTestImgsPerArtist;

trainImgs = cell(nTrainImgs, 1);
trainLbls = zeros(nTrainImgs, 1);
testImgs = cell(nTestImgs, 1);
testLbls = zeros(nTestImgs, 1);
trn_i = 1;
tst_i = 1;
for L = 1:nLbls
    labelPath = fullfile(paintPath, labelNames{L});
    labelDir = dir(fullfile(labelPath, '*'));
    artistNames = {labelDir(3:end).name}; % excludes . and .. (current and parent dirs)
    for a = 1:nArtistsPerLbl
        artistPath = fullfile(labelPath, artistNames{a});
        artistDir = dir(fullfile(artistPath, ['*.' imgNameExt]));
        imageNames = {artistDir(1:nImgsPerArtist).name};
        % training images
        for i = 1:nTrainImgsPerArtist
            img = imread(fullfile(artistPath, imageNames{i}));
            img = imfilter(img, gaussFilt, 'symmetric');
            trainImgs{trn_i} = img;
            trainLbls(trn_i) = L;
            trn_i = trn_i + 1;
        end
        % testing images
        for i = nTrainImgsPerArtist+1:nImgsPerArtist
            img = imread(fullfile(artistPath, imageNames{i}));
            img = imfilter(img, gaussFilt, 'symmetric');
            testImgs{tst_i} = img;
            testLbls(tst_i) = L;
            tst_i = tst_i + 1;
        end
    end
end
