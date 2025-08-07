%%% Sample Investigation for Static/Driving Datasets
%%% Can be used to obtain Fig. 1 and Fig. 2 (Ground Truth) in XAI in Wireless Communications: A Case Study on Interpretable 5G Performance Analysis

clc;
clear;
close all;
load("Datasets/drivingData.mat");
drivingTable = finalTable;
drivingMatrix = finalMatrix;
load("Datasets/staticData.mat");
staticTable = finalTable;
staticMatrix = finalMatrix;

%% Data Investigation

figure;
subplot(2,2,1);
histogram(drivingTable.RSRP, 30);  % 30 bins, for example
set(gca, 'XDir', 'reverse');
title('Histogram of RSRP Values (Driving Case)');
xlabel('RSRP');
ylabel('Frequency');
grid on;

subplot(2,2,2);
plot(drivingTable.RSRP, '-o', 'DisplayName','RSRP');
hold on;
mu = mean(drivingTable.RSRP);
hLine = yline(mu, '--r', 'LineWidth', 3,'DisplayName', sprintf('Mean RSRP: %.2f dBm', mu)); %#ok<*NASGU>
title('Ground Truth: RSRP Values (Driving Case)');
xlabel('Sample Index');
ylabel('RSRP');
drivingSample = size(drivingMatrix,1);
xlim([0 drivingSample]);
ylim([-130 -30])
grid on;
legend('Location','northeast',Box='off'); 

subplot(2,2,3);
histogram(staticTable.RSRP, 30);  % 30 bins, for example
set(gca, 'XDir', 'reverse');
title('Histogram of RSRP Values (Static Case)');
xlabel('RSRP');
ylabel('Frequency');
grid on;

subplot(2,2,4);
plot(staticTable.RSRP, '-o', 'DisplayName','RSRP');
hold on;
mu = mean(staticTable.RSRP);
hLine = yline(mu, '--r', 'LineWidth', 3,'DisplayName', sprintf('Mean RSRP: %.2f dBm', mu));
title('Ground Truth: RSRP Values (Static Case)');
xlabel('Sample Index');
ylabel('RSRP');
staticSample = size(staticMatrix,1);
xlim([0 staticSample]);
grid on;
legend('Location','northeast',Box='off'); 

set(gcf, 'Position', [100, 100, 1200, 800])
exportgraphics(gcf, 'sampleAnalysis.eps', 'ContentType', 'vector')

sgtitle("RSRP Sample Investigation for Both Datasets");

%% Ground Truth Visualization

figure('WindowState','maximized');
tiledlayout(1, 6, 'Padding', 'none', 'TileSpacing', 'compact'); 

numBins = 250; % Number of bins for averaging
featuresOfInterest = ["RSSI", "SNR", "RSRQ", "CQI"];

for i = 1:numel(featuresOfInterest)
    ax = nexttile; % Use nexttile instead of subplot
    xData = drivingTable.(featuresOfInterest(i));
    yData = drivingTable.RSRP;

    scatter(xData, yData, 10, yData, 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;

    [binCounts, edges, binIdx] = histcounts(xData, numBins);
    binCenters = (edges(1:end-1) + edges(2:end)) / 2; 
    meanRSRP = accumarray(binIdx(~isnan(binIdx)), yData(~isnan(binIdx)), [], @mean, NaN);

    validBins = ~isnan(meanRSRP);
    binCenters = binCenters(validBins);
    meanRSRP = meanRSRP(validBins);

    plot(binCenters, meanRSRP, 'r-', 'LineWidth', 2);

    if strcmp(featuresOfInterest(i), 'SNR')  
        ylim([0.55, 1.0125]); 
    else
        yMin = min(meanRSRP) - 0.0125;
        yMax = max(meanRSRP) + 0.0125;
        ylim([yMin, yMax]);
    end

    xlabel(featuresOfInterest(i));
    ylabel('RSRP');
    title(featuresOfInterest(i));
    grid on;
    hold off;
    
    % Make the plot stretch within the tile
    set(ax, 'Position', get(ax, 'OuterPosition'));
end

axHeat1 = nexttile; % Use nexttile instead of subplot
xData = drivingTable.Longitude;
yData = drivingTable.RSSI;
zData = drivingTable.RSRP;

[xBins, yBins, heatmapData] = createHeatmap(xData, yData, zData, numBins);

imagesc(xBins, yBins, heatmapData);
set(gca, 'YDir', 'normal');
colormap(axHeat1, 'turbo');
xlabel('Longitude');
ylabel('RSSI');
title('Longitude & RSSI');

% Heatmap 2: Longitude & Latitude
axHeat2 = nexttile; % Use nexttile instead of subplot
xData = drivingTable.Longitude;
yData = drivingTable.Latitude;
zData = drivingTable.RSRP;

[xBins, yBins, heatmapData] = createHeatmap(xData, yData, zData, numBins);

imagesc(xBins, yBins, heatmapData);
set(gca, 'YDir', 'normal');
colormap(axHeat2, 'turbo');
xlabel('Longitude');
ylabel('Latitude');
title('Longitude & Latitude');

function [xCenters, yCenters, heatmapData] = createHeatmap(x, y, z, numBins)
    [binCounts, xEdges, yEdges, binX, binY] = histcounts2(x, y, numBins);
    heatmapData = accumarray([binX, binY], z, [numBins numBins], @mean, NaN);
    xCenters = (xEdges(1:end-1) + xEdges(2:end)) / 2;
    yCenters = (yEdges(1:end-1) + yEdges(2:end)) / 2;
end

