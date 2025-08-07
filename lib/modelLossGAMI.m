function [bceLoss,loss, firstGradients, secondGradients, netMain, netInt, numerNetsMain, numerNetsInt] = modelLossGAMI(netMain, netInt, X, T, numLayers, features, interactions, clarityParameter, numerNetsMain, trainingCheckMain, numerNetsInt, trainingCheckInt)
    
    fieldNames = fieldnames(netMain);
    isSubnet = startsWith(fieldNames, 'subnet');
    numFeatures = sum(isSubnet);    
    numInteractions = size(interactions,1);
    
    [~, Ymain, numerNetsMain] = model(netMain, X, numLayers, numFeatures, numerNetsMain, trainingCheckMain);
    Ymain = Ymain';
    
    [Yint, featureY, firstYint, secondYint, numerNetsInt] = modelInt(netInt, netMain, X, numLayers, features, interactions, numerNetsMain, trainingCheckMain, numerNetsInt, trainingCheckInt);
    Yint = Yint';
    tempClarityLoss = zeros(1,numInteractions);
    
    for ii = 1:numInteractions
        if ismember(interactions(ii, 1), features) && ismember(interactions(ii, 2), features)
            b = featureY(:, ii);
            a1 = firstYint(:, ii);
            a2 = secondYint(:, ii);

            firstTerm = abs(mean(b.*a1));
            secondTerm = abs(mean(b.*a2));
            tempClarityLoss(ii) = firstTerm + secondTerm;
        elseif ismember(interactions(ii, 1), features)
            b = featureY(:, ii);
            a1 = firstYint(:, ii);
            tempClarityLoss(ii) = abs(mean(b.*a1));
        else
            b = featureY(:, ii);
            a2 = secondYint(:,ii);
            tempClarityLoss(ii) = abs(mean(b.*a2));
        end

    end
    
    Y = Ymain + Yint;
    clarityLoss = clarityParameter*sum(tempClarityLoss);
    
    epsVal = 1e-6;
    pred = Y;
    pred = min(max(pred, epsVal), 1 - epsVal);
    crossEntropy = - (T .* log(pred) + (1 - T) .* log(1 - pred));
    bceLoss = mean(crossEntropy);

    loss = bceLoss + clarityLoss;
    
    firstGradients = dlgradient(bceLoss, netMain);
    secondGradients = dlgradient(loss, netInt);
    firstGradients.mainEffectSwitcher = dlarray(zeros(numFeatures,1));
    secondGradients.interactionSwitcher = dlarray(zeros(numInteractions,1));
  
    end