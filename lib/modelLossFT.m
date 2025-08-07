function [loss,gradients,Y,numerNets,numerNetsInt] = modelLossFT(netMain, netInt, X, T, numLayers, features, interactions, numerNets, trainingCheck, numerNetsInt, trainingCheckInt)
    
    fieldNames = fieldnames(netMain);
    isSubnet = startsWith(fieldNames, 'subnet');
    numFeatures = sum(isSubnet);

    [~, Ymain, ~] = model(netMain, X, numLayers, numFeatures, numerNets, trainingCheck);
    Ymain = Ymain';

    [Yint, ~, ~, ~, numerNetsInt] = modelInt(netInt, netMain, X, numLayers, features, interactions, numerNets, trainingCheck, numerNetsInt, trainingCheckInt);
    Yint = Yint';

    Y = Ymain + Yint;
    
    epsVal = 1e-6;
    pred = Y;
    pred = min(max(pred, epsVal), 1 - epsVal);
    crossEntropy = - (T .* log(pred) + (1 - T) .* log(1 - pred));
    loss = mean(crossEntropy);

    gradients = dlgradient(loss, netInt);
    
end