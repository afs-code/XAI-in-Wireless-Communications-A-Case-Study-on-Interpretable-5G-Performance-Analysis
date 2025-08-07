function [Y, featureY, firstComponentY, secondComponentY, numerNetsInt] = modelInt(net, netMain, X, numLayer, features, interactions, numerNets, trainingCheck,numerNetsInt, trainingCheckInt)
    
    numInteractions = size(interactions,1);
    
    Y = dlarray(zeros(1,size(X, 1)));
    featureY = dlarray(zeros(numInteractions, size(X, 1)));
    firstComponentY = dlarray(zeros(numInteractions, size(X, 1)));
    secondComponentY = dlarray(zeros(numInteractions, size(X, 1)));

    for idx = 1:numInteractions
        xTemp = X(:, interactions(idx, :));
        firstXComponent = X(:, interactions(idx, 1));
        secondXComponent = X(:, interactions(idx, 2));

        [subPred,numerNetsInt] = submodelInt(net.("subnet" + idx), xTemp, numLayer, numerNetsInt, trainingCheckInt, idx);

        if ismember(interactions(idx, 1), features)
            [firstSubPred, ~] = submodel(netMain.("subnet" + interactions(idx, 1)), firstXComponent, numLayer, numerNets, trainingCheck, interactions(idx, 1));
        else
            firstSubPred = dlarray(zeros(1, size(firstXComponent, 1)));
        end
        if ismember(interactions(idx, 2), features)
            [secondSubPred, ~] = submodel(netMain.("subnet" + interactions(idx, 2)), secondXComponent, numLayer, numerNets, trainingCheck, interactions(idx, 2));
        else
            secondSubPred = dlarray(zeros(1, size(firstXComponent, 1)));
        end
        
        featureY(idx,:) = subPred;
        firstComponentY(idx,:) = firstSubPred;
        secondComponentY(idx,:) = secondSubPred;
    end

    Y = net.outputLayer.Weights*(featureY.*net.interactionSwitcher) + net.outputLayer.Bias;
    featureY = featureY';
    firstComponentY = firstComponentY';
    secondComponentY = secondComponentY';

end