function [loss, yPred, numerNets] = evaluate(net, X, T, numLayers, numFeatures, numerNets, trainingCheck)
    
    [loss, ~, ~, yPred, numerNets] = dlfeval(@modelLoss, net, X, T, numLayers, numFeatures, numerNets, trainingCheck);
    
end