function [colorHists] = getRgbHistsBlock(images)
%GETRGBHISTSBLOCK Calculates a color histogram for each image
%   Note: The color histogram is a concatenation of the histograms of each 
%   of the RGB color channels.

global nHistBins imgBlocks
nImgs = length(images);
nBlocks = prod(imgBlocks);

colorHists = zeros(nImgs, 3 * nHistBins * nBlocks);
for i = 1:nImgs
    img = images{i};
    rStep = floor(size(img,1) ./ imgBlocks(1));
    cStep = floor(size(img,2) ./ imgBlocks(2));
    bStep = nHistBins;
    if size(img, 3) == 3
        redHist = zeros(nHistBins * nBlocks, 1);
        grnHist = zeros(nHistBins * nBlocks, 1);
        bluHist = zeros(nHistBins * nBlocks, 1);
        b = 1;
        for br = 1:imgBlocks(1)
            rTmp = (br-1)*rStep+1 : br*rStep;
            for bc = 1:imgBlocks(2)
                cTmp = (bc-1)*cStep+1 : bc*cStep;
                bTmp = (b-1)*bStep+1 : b*bStep;
                redHist(bTmp) = imhist(img(rTmp, cTmp, 1), nHistBins);
                grnHist(bTmp) = imhist(img(rTmp, cTmp, 2), nHistBins);
                bluHist(bTmp) = imhist(img(rTmp, cTmp, 3), nHistBins);
                b = b + 1;
            end
        end
        colorHists(i, :) = [redHist; grnHist; bluHist]';
    else
        grayHist = zeros(nHistBins * nBlocks, 1);
        b = 1;
        for br = 1:imgBlocks(1)
            rTmp = (br-1)*rStep+1 : br*rStep;
            for bc = 1:imgBlocks(2)
                cTmp = (bc-1)*cStep+1 : bc*cStep;
                bTmp = (b-1)*bStep+1 : b*bStep;
                grayHist(bTmp) = imhist(img(rTmp, cTmp), nHistBins);
                b = b + 1;
            end
        end
        colorHists(i, :) = [grayHist; grayHist; grayHist]';
    end
    tmp = colorHists(i, :);
    colorHists(i, :) = colorHists(i, :) / sum(tmp(:));
end
