function [Y, numerNets] = submodelInt(subnet, X, numLayers, numerNets, trainingCheck, ii)

    if trainingCheck == "true"
        X = X';
        for ct=1:numLayers-1
            X = subnet.("fc"+ct).Weights*X + subnet.("fc"+ct).Bias;
            X = relu(X);
        end
        Y = subnet.("outputLayer").Weights*X + subnet.("outputLayer").Bias;
        subnetMean = mean(Y);
        subnetNorm = var(Y);
        numerNets.("numerNet" + ii).subnetMean = subnetMean;
        numerNets.("numerNet" + ii).subnetNorm = subnetNorm;
        numerNets.("numerNet" + ii).movingMean = numerNets.("numerNet" + ii).subnetMean;
        numerNets.("numerNet" + ii).movingNorm = numerNets.("numerNet" + ii).subnetNorm;
    else
        X = X';
        for ct=1:numLayers-1
            X = subnet.("fc"+ct).Weights*X + subnet.("fc"+ct).Bias;
            X = relu(X);
        end
        Y = subnet.("outputLayer").Weights*X + subnet.("outputLayer").Bias;
        numerNets.("numerNet" + ii).subnetMean = numerNets.("numerNet" + ii).movingMean;
        numerNets.("numerNet" + ii).subnetNorm = numerNets.("numerNet" + ii).movingNorm;
    end

end