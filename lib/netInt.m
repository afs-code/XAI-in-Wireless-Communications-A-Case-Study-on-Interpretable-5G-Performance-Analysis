function [nets, numerNets] = netInt(numNodes, numLayers, numInput, numOutput, numNetworks, allIdx, activeIdx)

    nets = struct;

    for i = 1:numNetworks
        nets.("subnet" + i) = struct;
        nets.("subnet" + i) = subnet(numNodes, numLayers, numInput, numOutput);
    end
    
    nets.("outputLayer") = struct;
    sz = [1, numNetworks];
    nets.("outputLayer").Weights = initializeGlorot(sz, 1, numNetworks);
    nets.("outputLayer").Bias = initializeZeros([1 1]);
    
    nets.("interactionSwitcher") = dlarray(zeros(numNetworks, 1));
    
    for ii = 1:numNetworks
        if ismember(allIdx(ii,:),activeIdx)
            nets.("interactionSwitcher")(ii) = dlarray(1);
        else
            nets.("interactionSwitcher")(ii) = dlarray(0);
        end
        numerNets.("numerNet" + ii).subnetMean = 0;
        numerNets.("numerNet" + ii).subnetNorm = 0;
        numerNets.("numerNet" + ii).movingMean = 0;
        numerNets.("numerNet" + ii).movingNorm = 0;
    end

end