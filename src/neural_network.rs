use ndarray::Array2;

use crate::layer::Layer;

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self {
            layers
        }
    }

    pub fn forward_propagate(&self, inputs: Vec<f32>) -> Vec<Array2<f32>> {

        // Turn our inputs into a matrix of the same size to fit our network's input shape (assume length matches)
        let inputs_array = Array2::from_shape_vec((inputs.len(), 1), inputs).unwrap();
        
        // Track each layer, here activation functions are also considered their own layer
        let mut forward_passes: Vec<Array2<f32>> = Vec::new();
        for layer_i in 0..self.layers.len() - 1 {
            
            // We're going to get our last layer's output
            let previous_pass= forward_passes.get(layer_i.saturating_sub(1)).unwrap_or(&inputs_array);

            // And feed it into the next layer
            let transfers = self.layers[layer_i].forward(previous_pass);

            // Save it here as the input for the next layer
            forward_passes.push(transfers);
        }

        forward_passes
    }

    pub fn backward(&self, activation_layers: &[Array2<f32>], target: &Array2<f32>) -> Vec<Array2<f32>> {
        
        // Define our gradients for each layer
        let mut gradients = Vec::new();

        // Caluclate the error at the output layer
        let final_output = activation_layers.last().unwrap();
        let error = final_output - target;

        // Compute the gradients for the output layer
        // let output_gradient = &error * final_output.mapv();
        // gradients.push(output_gradient.clone());

        // Propagate backward through the hidden layers
        let mut last_gradient = output_gradient; 
        for layer_i in (0..activation_layers.len() - 1).rev() {
            let activations = &activation_layers[layer_i];
            // let weights = &self.layers[layer_i].weights;

            
        }

        vec![]
    }

    pub fn backwards_propagate(&mut self, activation_layers: &[Array2<f32>], target: &Array2<f32>) -> Vec<Array2<f32>> {


        // Backpropagate through the hidden layers
        let mut last_gradient = output_gradient;
        for layer_i in (0..activation_layers.len() - 1).rev() {
            let activations = &activation_layers[layer_i];
            let weights = &self.weight_layers[layer_i];

            // Compute gradients for the current hidden layer
            
            // Step 1: transpose the weights
            let transposed_weights = weights.t();

            // Step 2: Propagate the error backwards
            let propagated_error = last_gradient.dot(&transposed_weights);

            // Step 3: Compute the derivative of the activation function
            let activation_derivaties = activations.mapv(relu_derivative);

            // Step 4: Multiply each propagated error and activation derivative
            let layer_gradient = propagated_error * activation_derivaties;
            
            gradients.push(layer_gradient.clone());

            // Update last_gradient for the next iteration
            last_gradient = layer_gradient;
        }

        // Reverse gradients so they match layer order (input -> output)
        gradients.reverse();
        gradients
    }

}
