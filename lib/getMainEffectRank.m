function [sortedIdx, componentScales] = getMainEffectRank(net, numFeatures, numerNets)
    
    mainEffectNorms = zeros(1,numFeatures);

    for i = 1:numFeatures
        mainEffectNorms(1, i) = numerNets.("numerNet" + i).movingNorm;
    end

    beta = net.outputLayer.Weights.^2.*mainEffectNorms;
    componentScales = abs(beta)./sum(abs(beta));
    componentScales = extractdata(componentScales);
    [~, sortedIdx] = sort(componentScales);
    sortedIdx = flip(sortedIdx);
    
end