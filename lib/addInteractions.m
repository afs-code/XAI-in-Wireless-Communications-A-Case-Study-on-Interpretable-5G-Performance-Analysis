function [topKPairs, interactionScores, interactionPairs] = addInteractions(X, Y, numInteractions, activeMainEffectsIndex, Heredity)
    
    % Add your path to interpretML environment here!

    % pyenv('Version', 'C:\Users\***\miniconda3\envs\interpretML\python', 'ExecutionMode', 'OutOfProcess');
    
    % Add your path to interpretML code here!
    % insert(py.sys.path,int32(0),'C:\Users\***\interpret');
    
    getInteractionScores = py.importlib.import_module('getIntScores');
    
    pythonX = double(extractdata(X));
    pythonY = double(extractdata(Y));

    numpyX = py.numpy.array(pythonX, pyargs('dtype', 'float64'));
    numpyY = py.numpy.array(pythonY, pyargs('dtype', 'float64'));
    
    interactionScoresPython = getInteractionScores.get_interaction_scores(numpyX,numpyY);

    topKPairs = zeros(numInteractions, 2);
    interactionScores = zeros(numInteractions, 1);

    % Select top-K interactions
    for i = 1:numInteractions
        pair = interactionScoresPython{i}{1};
        score = interactionScoresPython{i}{2};

        topKPairs(i, :) = [double(pair{1}), double(pair{2})];  
        interactionScores(i) = double(score);
    end

    py.importlib.reload(getInteractionScores);
    
    interactionPairs = [];

    for i = 1:numInteractions
        pair = topKPairs(i, :);
        if Heredity  % Strong heredity case
            if ismember(pair(1), activeMainEffectsIndex) && ismember(pair(2), activeMainEffectsIndex)
                interactionPairs = [interactionPairs; pair];
            end
        else        % Weak heredity case
            if ismember(pair(1), activeMainEffectsIndex) || ismember(pair(2), activeMainEffectsIndex)
                interactionPairs = [interactionPairs; pair];
            end
        end
    end

end
