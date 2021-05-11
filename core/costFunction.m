function [costValue] = costFunction(yTrain,yTarget,outputLayerNumber,index)

costValueInt = 0;

    for k = 1 : 1 : outputLayerNumber

        costValueInt = costValueInt + 1/2*(yTarget(k,index) - yTrain(k,index))*(yTarget(k,index) - yTrain(k,index));
    
    end
    
costValue = costValueInt;

end