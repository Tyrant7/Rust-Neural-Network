use ndarray::Array2;

use crate::layer::{self, Layer};

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self {
            layers
        }
    }

    pub fn forward(&self, inputs: &Array2<f32>) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {

        // Track each layer of activations through the network, this is what we'll be returning
        // In this case, the last layer of activations represents the network's output
        let mut activations: Vec<Array2<f32>> = Vec::new();
        let mut transfers: Vec<Array2<f32>> = Vec::new();
        
        for layer_i in 0..self.layers.len() {

            // Get the input to the current layer, whatever came last
            let previous_activations: &Array2<f32> = match (layer_i as i32 - 1) < 0 {
                true => inputs,
                false => &activations[layer_i - 1]
            };
            let layer = &self.layers[layer_i];

            // Forward through the current layer
            let transfer_layer = layer.forward(previous_activations); // but without activation
            transfers.push(transfer_layer.clone());
            let layer_activations = layer.activate(transfer_layer);

            // Push to the stack for next layer
            activations.push(layer_activations);
        }

        (activations, transfers)
    }

    pub fn backwards(&mut self, activations: &[Array2<f32>], transfers: &[Array2<f32>], inputs_count: f32, targets: Vec<f32>) -> Vec<(Array2<f32>, Array2<f32>)> {
        
        // Define our gradients for each layer, this is what we'll be returning
        let mut gradients = Vec::new();

        // Calculate the error at the output layer
        let final_output = activations.last().unwrap();

        // Construct our targets array, assume shape matches network output shape
        let targets_array = Array2::from_shape_vec((targets.len(), 1), targets).unwrap();

        // TODO: loss functions
        let error = final_output - targets_array;
        println!("Error: {}", error);

        // let error = -(&targets_array / final_output - (1. - targets_array.clone()) / (1. - targets_array.clone()));

        // Propagate backwards over all layers, skipping the input layer
        let mut output_gradient = error;

        for layer_i in (0..self.layers.len()).rev() {
            let layer = &self.layers[layer_i];
            // Check we underflow (to determine if we are on the input layer)
            
            let previous_transfers = &transfers[layer_i];
            let layer_activations = &activations[layer_i];

            let (new_gradient, weight_gradient, bias_gradient) = layer.backward(layer_activations, previous_transfers, &output_gradient);
            output_gradient = new_gradient;

            /* println!("weights shape {:?}", weight_gradient.shape());
            println!("bias shape {:?}", bias_gradient.shape()); */

            gradients.push((weight_gradient / inputs_count, bias_gradient / inputs_count));
        }

        // Reverse gradients so they match layer order (input -> output)
        gradients.reverse();
        gradients
    }
}
