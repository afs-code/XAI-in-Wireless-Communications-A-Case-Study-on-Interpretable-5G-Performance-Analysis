function [loss, yPred, numerNetsInt] = evaluateInt(net, netMain, X, T, numLayers, features, interactions, clarityParameter, numerNets, trainingCheck, numerNetsInt, trainingCheckInt)
    
    [loss, ~, yPred, numerNetsInt] = dlfeval(@modelLossInt, net, netMain, X, T, numLayers, features, interactions, clarityParameter, numerNets, trainingCheck, numerNetsInt, trainingCheckInt);
    
end