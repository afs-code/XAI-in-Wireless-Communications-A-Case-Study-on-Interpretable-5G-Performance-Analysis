function [sortedIdx, componentScales] = getRank(net, numNetworks, numerNets, networkType)
    
    norms = zeros(1,numNetworks);

    for i = 1:numNetworks
        norms(1, i) = numerNets.("numerNet" + i).movingNorm;
    end

    if networkType == "mainEffect"
        beta = net.outputLayer.Weights.^2.*norms;
        componentScales = abs(beta)./sum(abs(beta));
    elseif networkType == "interaction"
        gamma = net.outputLayer.Weights(1:numNetworks).^2.*norms;
        componentScales = abs(gamma)./sum(abs(gamma));
    else
        error('\nWrong type of network has been given!\n Correct types: mainEffect, interaction.');
    end
    componentScales = extractdata(componentScales);
    [~, sortedIdx] = sort(componentScales);
    sortedIdx = flip(sortedIdx);
    
end