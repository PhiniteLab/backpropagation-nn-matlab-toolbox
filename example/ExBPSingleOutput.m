%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Standard Backpropagation Algorithm

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% set path to core for libraries
addpath(genpath('../core'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% creating dataset


%% Writing the dataset on the graph

t = [0:0.01:1];

trainingNumber = length(t(1,:));

%% for generalized assignment to X and Y


% input vector
X0 = ones(1,trainingNumber);    % for bias input
X1 = t(1,:);                    % for first input
xGeneralInput = [X0;X1];


% output vector
yGeneralTarget = [sin(2*pi*1*X1)];             % target values

%% creating dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NN parameters

number_of_input_layer_node = length(xGeneralInput(:,1));
number_of_hidden_layer_node = 8;
number_of_output_layer_node = length(yGeneralTarget(:,1));


I = number_of_input_layer_node;
H = number_of_hidden_layer_node;
K = number_of_output_layer_node;

%% NN parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% optimization criteria for solving nonlinear equation

%% boundary conditions
errorMax = 0.000001;
iterationMax = 500000;


%% internal training parameters
learningRate = 0.1;


%% optimization criteria for solving nonlinear equation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% initialize the parameter required for training session

%% scalar
iteration = 0;

%% matrices
errorNow = ones(trainingNumber,K);
errorPre = ones(trainingNumber,K);

errorNowValue = sum(sum(errorNow))/trainingNumber;
errorPreValue = sum(sum(errorPre))/trainingNumber;

%% NN structure initialization
[W,W_previous,V,V_previous] = initializeNeuralNetwork(H, K, I);

d_W = zeros(size(W));
d_V = zeros(size(V));

[z,y] = creatingActivationFunction(H, K, trainingNumber);

%% initialize the parameter required for training session
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% general information related to the process

disp('Neural Network is started!');
disp('Basic information can be given by...');

disp('  ')
displayMessage = ['Input Layer Node: ',num2str(I), ' Hidden Layer Node :', num2str(H),...
    ' Output Layer Node: ',num2str(K)];
disp(displayMessage)

disp('  ')
disp('Training neural network is started in five seconds')

disp('  ')
pause(1);

%% general information related to the process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% For training process

%% comparing with the total error
while (errorNowValue > errorMax) && (iteration < iterationMax)

    iteration = iteration + 1;
    errorPre = errorNow;

    %% initialize all delta values after the training session
    d_W = zeros(H,I);
    d_V = zeros(K,H+1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %% training session is started

    for i = 1 : 1 : trainingNumber

        % z value calculation
        z(:,i) = actFuncCalc(xGeneralInput,W,H,I,i);

        % y value calculation
        y(:,i) = outputFuncCalc(z,V,K,H,i);

        % delta v coefficient calculation
        d_V = updatingGradientV(y, yGeneralTarget, z ,d_V,learningRate,K,H,i);

        % delta w coefficient calculation
        d_W = updatingGradientW(y, yGeneralTarget, z, xGeneralInput,V, d_W, learningRate, K,H,I,i);

        for k = 1 : 1 : K
            
            errorNow(i,k) = costFunction(y,yGeneralTarget,K,i);

        end
        
    end

    %% training session is started
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    errorNowValue = sum(sum(errorNow))/trainingNumber;
    errorPreValue = sum(sum(errorPre))/trainingNumber;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% updating new weights for W and V
    V = V + d_V/trainingNumber;
    W = W + d_W/trainingNumber;

    W_previous = W;
    V_previous = V;

    %% updating new weights for W and V
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %% learningRate change
%     if (errorPreValue - errorNowValue) > 0
% 
%         internalAssessmentMu = (errorPreValue - errorNowValue);
% 
%         if abs(internalAssessmentMu) > (1e-4/trainingNumber)
% 
%             learningRate = learningRate + learningRate*0.01;
% 
%         else
% 
%             learningRate = learningRate - learningRate*0.01;
% 
%         end
% 
%     end
%     
%     %% learningRate change
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    if (mod(iteration, 10) == 0)
        
        displayMessage = ['Error: ',num2str(errorNowValue),' Iteration: ',...
                num2str(iteration),' learningRate: ',num2str(learningRate)];
         
        disp(displayMessage)

    end
    
end

%% For training process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%% reinitialize the neural network for the extending the properties
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% For test process

z_test = zeros(size(z));
y_test = zeros(size(y));

for i = 1 : 1 : trainingNumber

    % z value calculation
    z_test(:,i) = actFuncCalc(xGeneralInput,W,H,I,i);

    % y value calculation

    y_test(:,i) = outputFuncCalc(z,V,K,H,i);
    
end

y_model_plot = y_test;
Y_plot = yGeneralTarget;

for k = 1 : 1 : K
    
    figure
    plot(Y_plot(k,:))
    hold on
    plot(y_model_plot(k,:))
    
end


%% For test process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Dynamic Neural Network Application
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%