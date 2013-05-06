%SETPARAMS Sets parameters as global variables
%   Detailed explanation goes here

global paintPath labelNames imgNameExt nArtistsPerLbl nTrainImgsPerArtist
global nTestImgsPerArtist smallSize nDsiftSteps vocabSize nHistBins
global imgBlocks edgeFilt nLbls labels A C I P R S nTrainImgsPerLbl
global nTestImgsPerLbl nTrainImgs nTestImgs gaussFilt

% Image loading parameters
paintPath = 'paintings';
labelNames = {
	'abstract';
	'cubism';
	'impressionism';
	'pointillism';
	'renaissance';
	'surrealism';
    };
A = 1;
C = 2;
I = 3;
P = 4;
R = 5;
S = 6;
imgNameExt = 'jpg';
nArtistsPerLbl = 5;
nTrainImgsPerArtist = 10;
nTestImgsPerArtist = 10;
sig = 0.5;  % default: 0.5
smallSize = [200 200];

% Feature extraction parameters
nDsiftSteps = 4; % default: 1
vocabSize = 200;
nHistBins = 64;
imgBlocks = -1;%[1 1];
edgeFilt = fspecial('log');

% --- COMPUTED PARAMETERS ---
nLbls = numel(labelNames);
labels = 1:nLbls;
nTrainImgsPerLbl = nArtistsPerLbl * nTrainImgsPerArtist;
nTestImgsPerLbl = nArtistsPerLbl * nTestImgsPerArtist;
nTrainImgs = nLbls * nTrainImgsPerLbl;
nTestImgs = nLbls * nTestImgsPerLbl;
filtSize = (floor(3*sig) * 2 + 1) * ones(2,1);
gaussFilt = fspecial('gaussian', filtSize, sig);

clear sig filtSize

% Creates directory for caching data
if ~exist('data', 'dir')
    mkdir('data')
end
