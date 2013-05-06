function [ vocab, wordHists ] = getVocab( nImgs, descrs, nFeatsPerImg )
%GETVOCAB Creates a vocabulary of SIFT feature descriptors
%   Detailed explanation goes here

global vocabSize

% Build vocab
[vocab, descrToWord] = vl_kmeans(descrs, vocabSize);

% Create a word histogram for each training image
% TODO use histc
wordHists = zeros(nImgs, vocabSize);
F = 1;
for i = 1:nImgs
    for f = 1:nFeatsPerImg{i}
        w = descrToWord(F);
        wordHists(i,w) = wordHists(i,w) + 1;
        F = F + 1;
    end
end
for i = 1:nImgs
    wordHists(i,:) = wordHists(i,:) / sum(wordHists(i,:));
end
