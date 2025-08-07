function Y = globalExplain(net, subnet, X, numLayers, ii)
    
    X = X';
    for ct=1:numLayers-1
        X = subnet.("fc"+ct).Weights*X + subnet.("fc"+ct).Bias;
        X = relu(X);
    end
    subnetY = subnet.("outputLayer").Weights*X + subnet.("outputLayer").Bias;
    Y = net.outputLayer.Weights(ii)*(subnetY);

end