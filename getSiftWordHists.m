function [ wordHists ] = getSiftWordHists( nImgs, vocab, descrsPerImg, nFeatsPerImg )
%SIFTWORDHISTS Creates a histogram of SIFT words for each image using a vocabulary
%   Detaied explanation goes here

global vocabSize

wordHists = zeros(nImgs, vocabSize);
for i = 1:nImgs
    for f = 1:nFeatsPerImg{i}
        descr = descrsPerImg{i}(:,f);
        minW = 0;
        minDist = realmax;
        for w = 1:vocabSize
            word = vocab(:,w);
            dist = norm(double(descr) - word); % Euclidean distance
            if dist < minDist
                minW = w;
                minDist = dist;
            end
        end
        wordHists(i,minW) = wordHists(i,minW) + 1;
    end        
    wordHists(i,:) = wordHists(i,:) / sum(wordHists(i,:));
end
