use ndarray::{Array2, Axis};
use rand::Rng;

use crate::{activation_functions::ActivationFunction, layer::Layer};

pub struct Linear {
    pub weights: Array2<f32>,
    pub bias: Array2<f32>,
    pub activation_function: ActivationFunction,
}

impl Linear {
    pub fn new(input_shape: usize, output_shape: usize, activation_function: ActivationFunction) -> Self {
        Linear {
            weights: Array2::from_elem((output_shape, input_shape), 0.),
            bias: Array2::from_elem((output_shape, 1), 0.),
            activation_function,
        }
    }

    pub fn new_from_rand(input_shape: usize, output_shape: usize, activation_function: ActivationFunction) -> Self {
        let mut rng = rand::thread_rng();
        Linear {
            weights: Array2::from_shape_fn((output_shape, input_shape), |(_i, _j)| rng.gen_range(-1., 1.)),
            bias: Array2::from_shape_fn((output_shape, 1), |(_i, _j)| rng.gen_range(-1., 1.)),
            activation_function,
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        self.activation_function.plain(self.weights.dot(input) + &self.bias)
    }

    pub fn backward(&self, 
        activations: &Array2<f32>,
        previous_activations: &Array2<f32>,
        delta: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>) {

        // Compute gradients for weights and biases, these will be passed as 'out' parameters
        
        let activated_deltas = delta * self.activation_function.derivative(activations.clone());
        let shape = previous_activations.shape()[1] as f32;

        let weight_gradient = (1. / shape) * activated_deltas.dot(&previous_activations.t());
        let bias_gradient = (1. / shape) * activated_deltas.sum_axis(Axis(1)).insert_axis(Axis(1));
        /* *weight_gradient = delta.dot(&activation.t());
        *bias_gradient = delta.sum_axis(Axis(1)).insert_axis(Axis(1)); */
        
        // Compute the input gradient to propagate backward
        (self.weights.t().dot(&activated_deltas), weight_gradient, bias_gradient)
    }

    pub fn get_params(&self) -> (&Array2<f32>, &Array2<f32>) {
        (&self.weights, &self.bias)
    }

    pub fn get_params_mut(&mut self) -> (&mut Array2<f32>, &mut Array2<f32>) {
        (&mut self.weights, &mut self.bias)
    } 
}

impl Layer {
    pub fn linear(input_size: usize, output_size: usize, activation_function: ActivationFunction) -> Self {
        Layer::Linear(Linear::new_from_rand(input_size, output_size, activation_function))
    }
}
