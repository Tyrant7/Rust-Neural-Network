use ndarray::{Array2, Axis, Shape};

use crate::layer::{ActivationFunction, Layer};

pub struct Linear {
    weights: Array2<f32>,
    bias: Array2<f32>,
    activation_function: ActivationFunction,
}

impl Linear {
    pub fn new(input_shape: usize, output_shape: usize, activation_function: ActivationFunction) -> Self {
        Linear {
            weights: Array2::from_elem((output_shape, input_shape), 0.),
            bias: Array2::from_elem((input_shape, 1), 0.),
            activation_function,
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        self.weights.dot(input) + &self.bias
    }

    pub fn backward(&self, activations: &Array2<f32>) -> Array2<f32> {

        // TODO: save the changes made here
        // currently weights are being passed in reverse through the network, 
        // but intermediate steps are not saved
        // we'll need to save these intermediate values somewhere for the optimizer to use

        // let weight_gradient = activations.dot(&input.t());
        // let bias_gradient = activations.sum_axis(Axis(1)).insert_axis(Axis(1))

        // ON SECOND THOUGHT
        // We'll return the input, output, and bias gradients and update using an optimizer
        // outside of this method in the network itself. I think that would work better

        activations.dot(&self.weights.t())
    }
}

impl Layer {
    pub fn linear(input_size: usize, output_size: usize, activation_function: ActivationFunction) -> Self {
        Layer::Linear(Linear::new(input_size, output_size, activation_function))
    }
}
