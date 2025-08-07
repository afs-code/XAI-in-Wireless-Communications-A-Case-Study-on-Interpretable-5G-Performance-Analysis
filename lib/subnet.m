function Learnables = subnet(numNodes, numLayers, numInput, numOutput)

    % Input Layer
    Learnable = struct;
    Learnable.("fc"+1) = struct;
    sz = [numNodes numInput];
    Learnable.("fc"+1).Weights = initializeGlorot(sz, numNodes, numInput);
    Learnable.("fc"+1).Bias = initializeZeros([numNodes 1]); 
    
    % Hidden Layers
    for ct = 2:numLayers-1

        Learnable.("fc"+ct) = struct;
        sz = [numNodes numNodes];
        Learnable.("fc"+ct).Weights = initializeGlorot(sz, numNodes, numNodes);
        Learnable.("fc"+ct).Bias = initializeZeros([numNodes 1]); 

    end

    % Output Layer
    Learnable.("outputLayer") = struct;
    sz = [numOutput numNodes];
    Learnable.("outputLayer").Weights = initializeGlorot(sz, numOutput, numNodes);
    Learnable.("outputLayer").Bias = initializeZeros([numOutput 1]); 

    Learnables = Learnable;

end