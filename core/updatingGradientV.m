function [d_V] = updatingGradientV(yAct, yTrain, zAct, weights, learningRate, outputLayerNumber, hiddenLayerNumber, index)
%% activation function

d_V_int = weights;

    for k = 1 : 1 : outputLayerNumber

        for h = 1 : 1 : (hiddenLayerNumber + 1)

            d_V_int(k,h) = d_V_int(k,h) + learningRate*(yTrain(k,index) - yAct(k,index))*zAct(h,index);

        end

    end

d_V = d_V_int;

end