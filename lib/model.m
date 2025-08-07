function [aggregatedY, Y, numerNets] = model(net, X, numLayer, numFeatures, numerNets, trainingCheck)
    
    aggregatedY = dlarray(zeros(1,size(X, 1)));
    Y = dlarray(zeros(numFeatures, size(X, 1)));
    for ct = 1:numFeatures
        [subPred, numerNets] = submodel(net.("subnet" + ct), X(:, ct), numLayer, numerNets, trainingCheck, ct);
        Y(ct,:) = subPred;
    end

    aggregatedY = sum(Y,1);
    aggregatedY = aggregatedY';

    Y = net.outputLayer.Weights*(Y.*net.mainEffectSwitcher) + net.outputLayer.Bias;

end