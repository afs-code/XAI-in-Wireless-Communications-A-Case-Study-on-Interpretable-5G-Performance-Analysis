function [loss, yPred, numerNetsMain, numerNetsInt] = evaluateFT(netMain, netInt, X, T, numLayers, features, interactions, numerNetsMain, trainingCheckMain, numerNetsInt, trainingCheckInt)
    
    [loss, ~, yPred, numerNetsMain, numerNetsInt] = dlfeval(@modelLossFT, netMain, netInt, X, T, numLayers, features, interactions, numerNetsMain, trainingCheckMain, numerNetsInt, trainingCheckInt);
    
end