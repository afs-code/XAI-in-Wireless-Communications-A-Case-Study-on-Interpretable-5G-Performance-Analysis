%%% GAMI-Net Implementation for Communication Dataset

clc;
clear;
close all;
addpath(genpath(fullfile(pwd, 'lib')));
load("Datasets/drivingData.mat");

%% Data Preparation
[dataNum, allFeatures] = size(finalMatrix);
numFeatures = allFeatures - 1; % Substract the target variable
testRatio = 0.2;
valRatio = 0.2;

normalize = "true";
featureNames = finalTable.Properties.VariableNames;

% Set the random seed for reproducibility
rng(0);

% Randomly chosen 10.000 samples
numSamples = 10^4;
rowIdx = randperm(dataNum, numSamples);
sampleMatrix = finalMatrix(rowIdx,:);

% 5G Dataset [2020]
x = sampleMatrix(:,(1:numFeatures));
y = sampleMatrix(:,end);

% Split the data into training and testing sets
cv = cvpartition(size(x, 1), 'HoldOut', testRatio);
idxTrainTest = training(cv);
idxTest = test(cv);

trainTestX = x(idxTrainTest, :);
trainTestY = y(idxTrainTest, :);
testX = dlarray(x(idxTest, :));
testY = dlarray(y(idxTest, :));

cvTrainVal = cvpartition(size(trainTestX, 1), 'HoldOut', valRatio);
idxTrain = training(cvTrainVal);
idxVal = test(cvTrainVal);

trainX = dlarray(trainTestX(idxTrain, :));
valX = dlarray(trainTestX(idxVal, :));
trainY = dlarray(trainTestY(idxTrain, :));
valY = dlarray(trainTestY(idxVal, :));

if normalize == "true"
    % Normalization after Train-Test Split
    valY = (valY - min(trainY)) / (max(trainY) - min(trainY));
    testY = (testY - min(trainY)) / (max(trainY) - min(trainY));
    trainY = (trainY - min(trainY)) / (max(trainY) - min(trainY));
end
%% GAMI-Net

% Model parameters
miniBatchSize = 200;
mainEffectEpochs = 500;
interactionEpochs = 500;
tuningEpochs = 500;
learningRate = [0.001, 0.001, 0.001];
earlyStopThreshold = [50 50 50];
lossThreshold = 0.01;
plotFreq = 50;
tolerance = 1e-5;

topKInteractions = 20;

Heredity = 0;   
marginalClarity = 0.0001;

averageGrad = [];
averageSqGrad = [];
averageGradInt = [];
averageSqGradInt = [];
gradDecay = 0.99;
sqGradDecay = 0.999;

minValLoss = inf;
minValLossInt = inf;
minValLossTune = inf;
estCounter = 0;
estCounterInt = 0; 
estCounterTune = 0;
numBatches = 0;
numValBatches = 0;

numNode = 40;
numLayer = 5;
numOutput = 1;

numInput = 1;
learnRate = learningRate(1);
networkType = "mainEffect";
[netMainEffects, numerNetsMainEffects] = net(numNode, numLayer, numInput, numOutput, numFeatures, networkType);

numIntInput = 2;
learnRateInt = learningRate(2);

learnRateTune = learningRate(3);

lossMainEffect = zeros(1,mainEffectEpochs);
lossInteraction = zeros(1,interactionEpochs);
lossFineTuning = zeros(1,tuningEpochs);

%% Training Main Effects

tic;
for epoch = 1:mainEffectEpochs

   shuffleIdx = randperm(size(trainX, 1));
   shuffleValIdx = randperm(size(valX, 1));

   for ct = 1:floor(size(trainX, 1)/miniBatchSize)
       idx = shuffleIdx((ct-1)*miniBatchSize+(1:miniBatchSize));
       miniBatchX = (trainX(idx, :));
       miniBatchY = (trainY(idx, :));

       mainEffectTraining = "true";

       [loss,gradients,~,~,numerNetsMainEffects] = dlfeval(@modelLoss, netMainEffects, miniBatchX, miniBatchY, numLayer, numFeatures, numerNetsMainEffects, mainEffectTraining);
       % Update network
       [netMainEffects,averageGrad,averageSqGrad] = adamupdate(netMainEffects,gradients,averageGrad,averageSqGrad,epoch,...
           learnRate,gradDecay,sqGradDecay);
       numBatches = numBatches + 1;
   end
    
   mainEffectTraining = "false";
   [trainLoss, ~, numerNetsMainEffects] = evaluate(netMainEffects, trainX, trainY, numLayer, numFeatures, numerNetsMainEffects, mainEffectTraining);
   [valLoss, ~, numerNetsMainEffects] = evaluate(netMainEffects, valX, valY, numLayer, numFeatures, numerNetsMainEffects, mainEffectTraining);

   lossMainEffect(epoch) = trainLoss;
   plotter(epoch, trainLoss)

   if valLoss < minValLoss       
       minValLoss = valLoss;
       estCounter = 0;
       bestModel = netMainEffects;
   else
       estCounter = estCounter + 1; % Increment patience counter
   end

   fprintf('Main Effect Epoch %d, Loss: %.4f, Validation Loss: %.4f, estCounter: %d\n', epoch, trainLoss, valLoss, estCounter);

   % Early stopping condition
   if estCounter >= earlyStopThreshold(1)
       disp('Stopping early due to no improvement in validation loss.');
       break;
   end

end
trainingTime = toc;
fprintf('Total training time of Main Effects: %.2f seconds\n', trainingTime);

%% Centering and Pruning the Main Effects && Add Interactions
 
mainEffectTraining = "true";
[trainLoss, ~, numerNetsMainEffects] = evaluate(bestModel, trainX, trainY, numLayer, numFeatures, numerNetsMainEffects, mainEffectTraining);
networkType = "mainEffect";
centeredNetMainEffects = center(bestModel,numerNetsMainEffects,numFeatures,networkType);

mainEffectTraining = "false";
[prunedNetMainEffects, activeMainEffectsIndex] = prune(centeredNetMainEffects, valX, valY, numFeatures, numLayer, numerNetsMainEffects, mainEffectTraining, lossThreshold);

[topKPairs, interactionScores, interactionPairs] = addInteractions(trainX, trainY, topKInteractions, activeMainEffectsIndex, Heredity);

numInteractions = length(interactionPairs);
networkType = "interaction";
[netInteractions, numerNetsInteractions] = net(numNode, numLayer, numIntInput, numOutput, numInteractions, networkType);

%% Training Interactions

tic;
for epochInt = 1:interactionEpochs

   shuffleIdx = randperm(size(trainX, 1));
   shuffleValIdx = randperm(size(valX, 1));

   for ct = 1:floor(size(trainX, 1)/miniBatchSize)
       idx = shuffleIdx((ct-1)*miniBatchSize+(1:miniBatchSize));
       miniBatchX = (trainX(idx, :));
       miniBatchY = (trainY(idx, :));
        
       interactionTraining = "true";

       [lossInt,gradientsInt,~,numerNetsInteractions] = dlfeval(@modelLossInt, netInteractions, prunedNetMainEffects, miniBatchX, miniBatchY, numLayer, activeMainEffectsIndex, interactionPairs, marginalClarity, numerNetsMainEffects, mainEffectTraining, numerNetsInteractions, interactionTraining);
       %Update network
       [netInteractions,averageGradInt,averageSqGradInt] = adamupdate(netInteractions,gradientsInt,averageGradInt,averageSqGradInt,epochInt,...
           learnRateInt,gradDecay,sqGradDecay);
       numBatches = numBatches + 1;
   end
   
   interactionTraining = "false";
   [trainLossInt, ~, numerNetsInteractions] = evaluateInt(netInteractions, prunedNetMainEffects, trainX, trainY, numLayer, activeMainEffectsIndex, interactionPairs, marginalClarity,numerNetsMainEffects, mainEffectTraining, numerNetsInteractions, interactionTraining);
   [valLossInt, ~, numerNetsInteractions] = evaluateInt(netInteractions, prunedNetMainEffects, valX, valY, numLayer, activeMainEffectsIndex, interactionPairs, marginalClarity, numerNetsMainEffects, mainEffectTraining, numerNetsInteractions, interactionTraining);

   lossInteraction(epochInt) = trainLossInt;
   plotter(epochInt, trainLossInt)

   if valLossInt < minValLossInt
       minValLossInt = valLossInt;
       estCounterInt = 0;
       bestModelInt = netInteractions;
   else
       estCounterInt = estCounterInt + 1; % Increment patience counter
   end

   fprintf('Interaction Epoch %d, Loss: %.4f, Validation Loss: %.4f, estCounter: %d\n', epochInt, trainLossInt, valLossInt, estCounterInt);

   % Early stopping condition
   if estCounterInt >= earlyStopThreshold(2)
       disp('Stopping early due to no improvement in validation loss.');
       break;
   end

end
trainingTimeInt = toc;
fprintf('Total training time of Interactions: %.2f seconds\n', trainingTimeInt);

%% Centering and Pruning the Interactions

interactionTraining = "true";
[trainLossInt, ~, numerNetsInteractions] = evaluateInt(bestModelInt, prunedNetMainEffects, trainX, trainY, numLayer, activeMainEffectsIndex, interactionPairs, marginalClarity,numerNetsMainEffects, mainEffectTraining, numerNetsInteractions, interactionTraining);
centeredNetInteractions = center(bestModelInt,numerNetsInteractions,numInteractions,networkType);

interactionTraining = "false";
[prunedNetInteractions, activeInteractionsIndex] = pruneInt(centeredNetInteractions, prunedNetMainEffects, valX, valY, numLayer, activeMainEffectsIndex, interactionPairs, marginalClarity, numerNetsMainEffects, mainEffectTraining, numerNetsInteractions, interactionTraining, lossThreshold);

%% Fine Tuning

tic;
for epochTune = 1:tuningEpochs

   shuffleIdx = randperm(size(trainX, 1));
   shuffleValIdx = randperm(size(valX, 1));

   for ct = 1:floor(size(trainX, 1)/miniBatchSize)
       idx = shuffleIdx((ct-1)*miniBatchSize+(1:miniBatchSize));
       miniBatchX = (trainX(idx, :));
       miniBatchY = (trainY(idx, :));
        
       mainEffectTraining = "true";
       interactionTraining = "true";

       [lossMainEffects, lossInteractions, gradients, gradientsInt, tunedNetMainEffects, tunedNetInteractions, numerNetsMainEffects, numerNetsInteractions] = dlfeval(@modelLossGAMI, prunedNetMainEffects, prunedNetInteractions, miniBatchX, miniBatchY, numLayer, activeMainEffectsIndex, interactionPairs, marginalClarity, numerNetsMainEffects, mainEffectTraining, numerNetsInteractions, interactionTraining);
       
       % Update both networks sequentially
       [tunedNetMainEffects,averageGrad,averageSqGrad] = adamupdate(tunedNetMainEffects,gradients,averageGrad,averageSqGrad,epochTune,...
           learnRateTune,gradDecay,sqGradDecay);
       [tunedNetInteractions,averageGradInt,averageSqGradInt] = adamupdate(tunedNetInteractions,gradientsInt,averageGradInt,averageSqGradInt,epochTune,...
           learnRateTune,gradDecay,sqGradDecay);

       numBatches = numBatches + 1;
   end
 
   mainEffectTraining = "false";
   interactionTraining = "false";
   [trainLossFT, ~, numerNetsMainEffects] = evaluateFT(tunedNetMainEffects, tunedNetInteractions, trainX, trainY, numLayer, activeMainEffectsIndex, interactionPairs,numerNetsMainEffects, mainEffectTraining, numerNetsInteractions, interactionTraining);
   [valLossFT, ~, numerNetsInteractions] = evaluateFT(tunedNetMainEffects, tunedNetInteractions, valX, valY, numLayer, activeMainEffectsIndex, interactionPairs, numerNetsMainEffects, mainEffectTraining, numerNetsInteractions, interactionTraining);
   
   lossFineTuning(epochTune) = trainLossFT;
   plotter(epochTune, trainLossFT)


   if valLossFT < minValLossTune
       minValLossTune = valLossFT;
       estCounterTune = 0;
       bestModelFineTunedMainEffect = tunedNetMainEffects;
       bestModelFineTunedInteraction = tunedNetInteractions;

   else
       estCounterTune = estCounterTune + 1; % Increment patience counter
   end

   fprintf('Fine Tuning Epoch %d, Loss: %.4f, Validation Loss: %.4f, estCounter: %d\n', epochTune, trainLossFT, valLossFT, estCounterTune);
   % Early stopping condition
   if estCounterTune >= earlyStopThreshold(3)
       disp('Stopping early due to no improvement in validation loss.');
       break;
   end

end
trainingTimeTune = toc;
fprintf('Total fine tuning time: %.2f seconds\n', trainingTimeTune);

%% Centering GAMI-Net

networkType = "mainEffect";
mainEffectTraining = "true";
[trainLossFTMain, ~, numerNetsMainEffects] = evaluate(bestModelFineTunedMainEffect, trainX, trainY, numLayer, numFeatures, numerNetsMainEffects, mainEffectTraining);
fineTunedNetMainEffects = center(bestModelFineTunedMainEffect,numerNetsMainEffects,numFeatures,networkType);
mainEffectTraining = "false";

networkType = "interaction";
interactionTraining = "true";
[trainLossFTInt, ~, numerNetsInteractions] = evaluateInt(bestModelFineTunedInteraction, fineTunedNetMainEffects, trainX, trainY, numLayer, activeMainEffectsIndex, interactionPairs, marginalClarity, numerNetsMainEffects, mainEffectTraining, numerNetsInteractions, interactionTraining);
fineTunedNetInteractions = center(bestModelFineTunedInteraction,numerNetsInteractions,numInteractions,networkType);
interactionTraining = "false";

%% Plotting the General Loss

mainEffectLossToCalc = lossMainEffect(lossMainEffect ~= 0);
interactionLossToCalc = lossInteraction(lossInteraction ~= 0);
fineTuningLossToCalc = lossFineTuning(lossFineTuning ~= 0);

totalLoss = [lossMainEffect(lossMainEffect ~= 0) lossInteraction(lossInteraction ~= 0) lossFineTuning(lossFineTuning ~= 0)];

lossMEIndex = 1:length(mainEffectLossToCalc);
lossIntIndex = (1:length(interactionLossToCalc)) + length(mainEffectLossToCalc);
lossFTIndex = (1:length(fineTuningLossToCalc)) + length(mainEffectLossToCalc) + length(interactionLossToCalc);

% Plot the total loss for each step in different colors
figure;
hold on;
plot(lossMEIndex, mainEffectLossToCalc, 'r', 'DisplayName', 'Main Effect','LineWidth', 1.5);
plot(lossIntIndex, interactionLossToCalc, 'g', 'DisplayName', 'Interaction','LineWidth', 1.5);
plot(lossFTIndex, fineTuningLossToCalc, 'b', 'DisplayName', 'Fine Tuning','LineWidth', 1.5);
grid on;
title('General Loss Values for Each Step');
legend('Main Effects','Interactions','Fine Tuning');
hold off;

%% RMSE Calculation

[trainMSE, ~,  numerNetsMainEffects, numerNetsInteractions] = evaluateFT(fineTunedNetMainEffects, fineTunedNetInteractions, trainX, trainY, numLayer, activeMainEffectsIndex, interactionPairs, numerNetsMainEffects, mainEffectTraining, numerNetsInteractions, interactionTraining);
trainRMSE = sqrt(trainMSE);
fprintf('Train RMSE: %.4f\n', mean(trainRMSE));

[testMSE, ~,  numerNetsMainEffects, numerNetsInteractions] = evaluateFT(fineTunedNetMainEffects, fineTunedNetInteractions, testX, testY, numLayer, activeMainEffectsIndex, interactionPairs, numerNetsMainEffects, mainEffectTraining, numerNetsInteractions, interactionTraining);
testRMSE = sqrt(testMSE);
fprintf('Test RMSE: %.4f\n', mean(testRMSE));

[valMSE, ~,  numerNetsMainEffects, numerNetsInteractions] = evaluateFT(fineTunedNetMainEffects, fineTunedNetInteractions, valX, valY, numLayer, activeMainEffectsIndex, interactionPairs, numerNetsMainEffects, mainEffectTraining, numerNetsInteractions, interactionTraining);
valRMSE = sqrt(valMSE);
fprintf('Validation RMSE: %.4f\n', mean(valRMSE));

totalTrainingTime = trainingTime + trainingTimeInt + trainingTimeTune;
fprintf('Total training time in all steps: %d minutes and %.2f seconds\n', floor(totalTrainingTime/60), rem(totalTrainingTime,60));

%% Global Interpretability & Active Features

gridSize = 100;
[xGrid, yGrid] = meshgrid(linspace(0, 1, gridSize), linspace(0, 1, gridSize));
inputGrid = [xGrid(:), yGrid(:)];

globalVisInput = transpose(linspace(0,1,gridSize));

numOfMainEffects = numel(activeMainEffectsIndex);
numOfInteractions = numel(activeInteractionsIndex);
numOfSubNetworks = numOfMainEffects + numOfInteractions;

if mod(numOfSubNetworks, 2) ~= 0
    totalPlots = numOfSubNetworks + 1; 
else
    totalPlots = numOfSubNetworks; 
end

figure;
for ii = 1:numOfSubNetworks
    if ii <= numOfMainEffects
        % Main effect
        featIdx = activeMainEffectsIndex(ii);              
        featName = featureNames{featIdx};                   

        subnet = fineTunedNetMainEffects.("subnet" + featIdx);
        outputVis = globalExplain(fineTunedNetMainEffects, subnet, globalVisInput, numLayer, featIdx);

        subplot(2, totalPlots/2, ii);
        plot(globalVisInput, outputVis, 'b', 'LineWidth', 1.5);
        grid on;
        title(['Main Effect: ' featName])                   
        xlabel('Normalized Input');                          
        ylabel('Output');
    else
        % Interaction
        idx = (ii - numOfMainEffects);
        intIdx1 = interactionPairs(idx, 1);
        intIdx2 = interactionPairs(idx, 2);
        featName1 = featureNames{intIdx1};                  
        featName2 = featureNames{intIdx2};                  

        subnet = fineTunedNetInteractions.("subnet" + activeInteractionsIndex(idx));
        outputGrid = globalExplain(fineTunedNetInteractions, subnet, inputGrid, numLayer, activeInteractionsIndex(idx));
        outputGrid = reshape(outputGrid, [gridSize, gridSize]);
        outputGrid = gather(extractdata(outputGrid));

        subplot(2, totalPlots/2, ii);
        mesh(xGrid, yGrid, outputGrid);
        view(2);
        colorbar;
        grid on;
        title(['Interaction Pair: ' featName1 ' & ' featName2])   
        xlabel(['Input 1: ' featName1]);
        ylabel(['Input 2: ' featName2]);
        zlabel('Output');
    end
end

sgtitle('Global Interpretability Results');


activeMainEffectNames = cell(numel(activeMainEffectsIndex),1);
for i = 1:numel(activeMainEffectsIndex)
    featIdx = activeMainEffectsIndex(i);
    activeMainEffectNames{i} = featureNames{featIdx};
end

activeInteractionNames = cell(numel(activeInteractionsIndex),1);
for i = 1:numel(activeInteractionsIndex)
    rowPair = interactionPairs(activeInteractionsIndex(i), :);  
    feat1Name = featureNames{rowPair(1)};
    feat2Name = featureNames{rowPair(2)};
    activeInteractionNames{i} = sprintf('%s & %s', feat1Name, feat2Name);
end

mainEffectsTable = table(activeMainEffectsIndex(:), activeMainEffectNames(:), ...
    'VariableNames', {'Index', 'FeatureName'});
disp('=== Active Main Effects Table ===');
disp(mainEffectsTable);

interactionPairIdx = interactionPairs(activeInteractionsIndex, :);
interactionsTable = table(interactionPairIdx(:,1), interactionPairIdx(:,2), activeInteractionNames(:), ...
    'VariableNames', {'Feat1Index','Feat2Index','InteractionName'});
disp('=== Active Interactions Table ===');
disp(interactionsTable);