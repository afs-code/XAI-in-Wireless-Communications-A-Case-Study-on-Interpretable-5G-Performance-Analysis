function [loss, gradients, aggregatedY, Y, numerNets] = modelLoss(net, X, T, numLayers, numFeatures, numerNets, trainingCheck)

    [aggregatedY, Y, numerNets] = model(net, X, numLayers, numFeatures, numerNets, trainingCheck);
    loss = sum((Y'-T).^2) /(size(X, 1));
    gradients = dlgradient(loss, net);
    gradients.mainEffectSwitcher = dlarray(zeros(numFeatures,1));

end