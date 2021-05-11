function [d_W] = updatingGradientW(yAct, yTrain, zAct, xInput, V_weights,d_W_weights, learningRate, outputLayerNumber, hiddenLayerNumber, inputLayerNumber, index)
%% activation function

d_W_int = d_W_weights;

    for h = 1 : 1 : hiddenLayerNumber
        
        for in = 1 : 1 : inputLayerNumber

            internalSum = 0.0;

            for k = 1 : 1 : outputLayerNumber

                internalSum = internalSum + (yTrain(k,index) - yAct(k,index))*V_weights(k,h);

            end

            d_W_int(h,in) = d_W_int(h,in) + learningRate*internalSum*zAct(h,index)*(1 - zAct(h,index))*xInput(in,index);

        end

    end

d_W = d_W_int;

end