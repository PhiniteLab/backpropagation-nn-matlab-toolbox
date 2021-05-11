function [yOutput] = outputFuncCalc(inputs,weights,outputLayerNumber,hiddenLayerNumber,index)
%% activation function

yOutInt = zeros(outputLayerNumber,1);

for k = 1 : 1 : outputLayerNumber
    
    internalSum = 0.0;
    
    for h = 1 : 1 : (hiddenLayerNumber + 1)
    
        internalSum = internalSum + weights(k,h)*inputs(h,index);
        
    end
    
    yOutInt(k,1) = internalSum;
    
end

yOutput = yOutInt;

end