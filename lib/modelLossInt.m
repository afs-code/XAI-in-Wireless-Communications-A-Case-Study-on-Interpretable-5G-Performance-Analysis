function [loss,gradients,Y,numerNetsInt] = modelLossInt(net, netMain, X, T, numLayers, features, interactions, clarityParameter, numerNets, trainingCheck, numerNetsInt, trainingCheckInt)
    
    fieldNames = fieldnames(netMain);
    isSubnet = startsWith(fieldNames, 'subnet');
    numFeatures = sum(isSubnet);
    numInteractions = size(interactions,1);

    [~, Ymain, ~] = model(netMain, X, numLayers, numFeatures, numerNets, trainingCheck);
    Ymain = Ymain';

    [Yint, featureY, firstYint, secondYint, numerNetsInt] = modelInt(net, netMain, X, numLayers, features, interactions, numerNets, trainingCheck, numerNetsInt, trainingCheckInt);
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
    mseLoss = sum((Y-T).^2) /(size(X, 1));
    loss = mseLoss + clarityLoss;

    gradients = dlgradient(loss, net);
    gradients.interactionSwitcher = dlarray(zeros(numInteractions,1));
    
    end