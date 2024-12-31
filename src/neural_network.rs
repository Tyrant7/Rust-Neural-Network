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

    pub fn forward(&self, inputs: Vec<f32>) -> Vec<Array2<f32>> {

        // Track each layer of activations through the network, this is what we'll be returning
        // In this case, the last layer of activations represents the network's output
        let mut activation_layers: Vec<Array2<f32>> = Vec::new();

        // Construct our input array, assume shape matches network shape
        let inputs_array = Array2::from_shape_vec((inputs.len(), 1), inputs).unwrap();

        for layer_i in 0..self.layers.len() - 1 {

            // Get the input to the current layer, whatever came last
            let previous_activations: &Array2<f32> = activation_layers.get(layer_i.saturating_sub(1)).unwrap_or(&inputs_array);

            // Forward through the current layer
            let activations = self.layers[layer_i].forward(previous_activations);

            // Push to the stack for next layer
            activation_layers.push(activations);
        }

        activation_layers
    }

    pub fn backwards(&mut self, activation_layers: &[Array2<f32>], target: &Array2<f32>) -> Vec<Array2<f32>> {

        // Define our gradients for each layer, this is what we'll be returning
        let mut gradients = Vec::new();

        // Calculate the error at the output layer
        let final_output = activation_layers.last().unwrap();
        let error = final_output - target;

        // Compute each gradient for the output layer
        let output_gradient = &error * self.layers.last().unwrap().backward(final_output);
        gradients.push(output_gradient);

        let mut last_gradient = output_gradient;
        for layer_i in (0..activation_layers.len() - 1).rev() {
            let activations = &activation_layers[layer_i];
            let layer = self.layers[layer_i];

            let propagated_error = layer.backward(&last_gradient);
            layer.backward(activations);
        }

        //


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
