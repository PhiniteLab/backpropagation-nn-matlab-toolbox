function [W,W_previous,V,V_previous] = initializeNeuralNetwork(number_of_hidden_layer_node, number_of_output_layer_node, number_of_input_layer_node)

    I = number_of_input_layer_node;
    
    H = number_of_hidden_layer_node;
    
    K = number_of_output_layer_node;

    W = randi([-100 100],H,I)./100;

    W_previous = randi([-100 100],H,I)./100;

    V = randi([-100 100],K,H+1)./100;

    V_previous = randi([-100 100],K,H+1)./100;


end