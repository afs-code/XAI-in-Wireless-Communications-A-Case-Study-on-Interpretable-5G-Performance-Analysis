%%% GAM Implementation for Communication Dataset

clc;
clear;
close all;
load("Datasets/drivingData.mat");

%% GAM
drivingTable = finalTable;
drivingMatrix = finalMatrix;
drivingTable.RSRP = (drivingTable.RSRP - min(drivingTable.RSRP)) / (max(drivingTable.RSRP) - min(drivingTable.RSRP));

% Driving table is chosen.
nSamples = 10000;  
rng(0);           
nTotal = height(drivingTable);
rowIdx = randperm(nTotal, min(nSamples, nTotal));
subTable = drivingTable(rowIdx, :);
allVarNames   = subTable.Properties.VariableNames;
predictorNames = setdiff(allVarNames, "RSRP");

cv = cvpartition(height(subTable), 'HoldOut', 0.2);  
trainData = subTable(training(cv), :);
testData     = subTable(test(cv), :);

trainPredictors = trainData(:, predictorNames);
trainResponse   = trainData.RSRP;
testPredictors  = testData(:, predictorNames);
testResponse    = testData.RSRP;

cvPartition = cvpartition(height(trainPredictors), 'HoldOut', 0.2);

disp('Fitting the GAM model...');
tic;  % Start timing the fit
gamModel = fitrgam(trainPredictors, trainResponse, ...
    'OptimizeHyperparameters','all-univariate',...
    'InitialLearnRateForPredictors', 0.001, ...
    'HyperparameterOptimizationOptions', struct( ... 
    'CVPartition', cvPartition, ...
    'MaxObjectiveEvaluations',500,...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'ShowPlots',false, 'Verbose',1,'UseParallel',true));

trainTime = toc;  
fprintf('GAM training took %.2f seconds.\n', trainTime);             

yPredTrain = predict(gamModel, trainPredictors);
trainMSE = mean((yPredTrain - trainResponse).^2);
trainRMSE = sqrt(trainMSE);
fprintf('Train MSE = %.4f, Train RMSE = %.4f\n', trainMSE, trainRMSE);

yPredTest = predict(gamModel, testPredictors);
testMSE = mean((yPredTest - testResponse).^2);
testRMSE = sqrt(testMSE);
fprintf('Test MSE = %.4f, Test RMSE = %.4f\n', testMSE, testRMSE);

cvGam   = crossval(gamModel, 'CVPartition', cvPartition);
valMSE = kfoldLoss(cvGam);
valRMSE = sqrt(valMSE);
fprintf('Validation MSE = %.4f, Validation RMSE = %.4f\n', valMSE, valRMSE);
 
%% Partial Dependence Plots (Main Effects)

nPreds = numel(predictorNames);
figure;
tic;
for i = 1:nPreds
    subplot(ceil(nPreds/2), 2, i);
    plotPartialDependence(gamModel, predictorNames{i});
    title(['Partial Dependence on ', predictorNames{i}]);
    grid on;
end
sgtitle('GAM Main Effects (Sampled)');
trainForPartialDependence = toc;
fprintf('Partial Dependence took %.2f seconds.\n', trainForPartialDependence);   

%% To obtain Fig. 2 (GAM) in XAI in Wireless Communications: A Case Study on Interpretable 5G Performance Analysis 
featuresOfInterest = ["RSSI", "SNR", "RSRQ", "CQI"];

figure('Units','normalized','OuterPosition',[0 0 1 1]);
tiledlayout(1, 4, 'Padding','none','TileSpacing','compact'); 

for i = 1:numel(featuresOfInterest)
    ax = nexttile; 
    plotPartialDependence(gamModel, featuresOfInterest(i));
    title(featuresOfInterest(i));
    grid on;

    set(ax, 'Position', get(ax, 'OuterPosition'));
end

