use ndarray::{Array2, Axis, Shape};

use crate::layer::Layer;
use super::activation_functions::ActivationFunction;

pub struct Linear {
    weights: Array2<f32>,
    bias: Array2<f32>,
    activation_function: ActivationFunction,
}

impl Linear {
    pub fn new(input_shape: usize, output_shape: usize, activation_function: ActivationFunction) -> Self {
        Linear {
            weights: Array2::from_elem((output_shape, input_shape), 0.),
            bias: Array2::from_elem((output_shape, 1), 0.),
            activation_function,
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        println!("About to multiply input with shape {:?} against weights with shape {:?}", input.shape(), self.weights.shape());

        let output = self.activation_function.plain(&(self.weights.dot(input) + &self.bias));

        println!("New shape: {:?}", output.shape());
        output
    }

    pub fn backward(&self, 
        activation: &Array2<f32>, 
        output_gradient: &Array2<f32>, 
        weight_gradient: &mut Array2<f32>, 
        bias_gradient: &mut Array2<f32>
    ) -> Array2<f32> {

        println!("input shape: {:?}", activation.shape());
        println!("output shape: {:?}", output_gradient.shape());

        println!("weights shape: {:?}", self.weights.shape());

        // Compute the derivative of the activation function then combine with the output gradient to get the delta
        let delta = output_gradient * self.activation_function.derivative(activation);

        println!("Delta shape: {:?}", delta.shape());

        // Compute gradients for weights and biases, these will be passed as 'out' parameters
        *weight_gradient = delta.dot(&activation.t());
        *bias_gradient = delta.sum_axis(Axis(1)).insert_axis(Axis(1));

        println!("Weight gradient shape: {:?}", weight_gradient.shape());

        println!("Delta with shape {:?} against weights with shape {:?}", delta.shape(), self.weights.t().shape());

        let output = self.weights.t().dot(&delta);

        println!("Output shape: {:?}", output.shape());

        // Compute the input gradient to propagate backward
        self.weights.t().dot(&delta)
    }
}

impl Layer {
    pub fn linear(input_size: usize, output_size: usize, activation_function: ActivationFunction) -> Self {
        Layer::Linear(Linear::new(input_size, output_size, activation_function))
    }
}
