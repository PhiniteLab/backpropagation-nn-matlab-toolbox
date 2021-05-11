function [act_output] = actFuncCalc(inputs, weights, hiddenLayerNumber, inputLayerNumber,index)
%% activation function

act_int = zeros(hiddenLayerNumber+1,1);

for h = 1 : 1 : (hiddenLayerNumber + 1)
    
    if h == (hiddenLayerNumber + 1)
       
        act_int(h,1) = 1;
        
    else
        
        internalSum = 0.0;
        
        for in = 1 : 1 : inputLayerNumber
            
            internalSum = internalSum + weights(h,in)*inputs(in,index);
        
        end
        
        act_int(h,1) = 1 / (1 + exp(-internalSum));
    end
    
end

act_output = act_int;

end