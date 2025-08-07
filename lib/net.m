function [nets, numerNets] = net(numNodes, numLayers, numInput, numOutput, numNetworks, networkType)

    nets = struct;

    for i = 1:numNetworks
        nets.("subnet" + i) = struct;
        nets.("subnet" + i) = subnet(numNodes, numLayers, numInput, numOutput);
    end
    
    nets.("outputLayer") = struct;
    sz = [1, numNetworks];
    nets.("outputLayer").Weights = initializeGlorot(sz, 1, numNetworks);
    nets.("outputLayer").Bias = initializeZeros([1 1]);
    if networkType == "mainEffect"
        nets.("mainEffectSwitcher") = dlarray(ones(numNetworks, 1));
    else
        nets.("interactionSwitcher") = dlarray(ones(numNetworks, 1));
    end
    for ii = 1:numNetworks
        numerNets.("numerNet" + ii).subnetMean = 0;
        numerNets.("numerNet" + ii).subnetNorm = 0;
        numerNets.("numerNet" + ii).movingMean = 0;
        numerNets.("numerNet" + ii).movingNorm = 0;
    end

end