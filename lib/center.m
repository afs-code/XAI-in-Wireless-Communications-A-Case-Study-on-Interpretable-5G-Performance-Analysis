function net = center(net, numerNets, numNetworks, networkType)

    outputBias = net.outputLayer.Bias;
    
    if networkType == "mainEffect"
        mainEffectWeights = net.mainEffectSwitcher.*net.outputLayer.Weights';
        for i = 1:numNetworks
            subnetBias = net.("subnet" + i).("outputLayer").Bias - numerNets.("numerNet" + i).movingMean;
            net.("subnet" + i).("outputLayer").Bias = subnetBias;
            outputBias = outputBias + numerNets.("numerNet" + i).movingMean.*mainEffectWeights(i);
        end

    elseif networkType == "interaction"  
        interactionWeights = net.interactionSwitcher.*net.outputLayer.Weights';
        for i = 1:numNetworks
            subnetBias = net.("subnet" + i).("outputLayer").Bias - numerNets.("numerNet" + i).movingMean;
            net.("subnet" + i).("outputLayer").Bias = subnetBias;
            outputBias = outputBias + numerNets.("numerNet" + i).movingMean.*interactionWeights(i);
        end

    else
        error('\nWrong type of network has been given!\n Correct types: mainEffect, interaction.');
    end

    net.outputLayer.Bias = outputBias;

end