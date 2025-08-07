function plotter(epoch,loss)
persistent hasRun
persistent start_time
persistent line_loss_train_persistent
persistent plotting_timesteps
persistent number_output
persistent oddVector;
persistent evenVector;

if epoch == 1
    hasRun = false;
    start_time = 0;
    line_loss_train_persistent = 0;
    plotting_timesteps = 0;
    oddVector = [];
    evenVector = [];

end

if hasRun == false

    start_time = tic;
    hasRun = true;

    for i=1:number_output*2
        if rem(i,2) == 1
            oddVector = [oddVector,i];
        else
            evenVector = [evenVector,i];
        end
    end

    f = figure;   
    C = colororder;
    line_loss_train_persistent = animatedline(Color=C(2,:));
    ylim([0 inf])
    xlabel("Epoch")
    ylabel("Loss")
    grid on

    
end

 %Plot Loss
    current_loss = double(loss);

    line_loss_train = line_loss_train_persistent;
%     line_loss_train = animatedline(Color=C(2,:));

    addpoints(line_loss_train,epoch,current_loss);
    D = duration(0,0,toc(start_time),Format="hh:mm:ss");
    title("Elapsed: " + string(D))
    drawnow
     % 

end