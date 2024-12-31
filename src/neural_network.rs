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
        let mut activation_layers: Vec<Array2<f32>> = Vec::new();
        for layer_i in 0..self.layers.len() - 1 {
            
            // We're going to get our last layer's output
            let previous_activations= activation_layers.get(layer_i.saturating_sub(1)).unwrap_or(&inputs_array);

            // And feed it into the next layer
            let transfers = self.layers[layer_i].forward(previous_activations);

            // Save it here as the input for the next layer
            activation_layers.push(transfers);
        }

        activation_layers
    }

}