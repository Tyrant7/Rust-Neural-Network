use ndarray::{linalg::Dot, Array2, Axis};
use rand::Rng;

use crate::{activation_functions::ActivationFunction, layer::Layer};

#[derive(Debug)]
pub struct Linear {
    pub weights: Array2<f32>,
    pub bias: Array2<f32>,
    pub activation_function: ActivationFunction,
}

impl Linear {
    pub fn new(input_shape: usize, output_shape: usize, activation_function: ActivationFunction) -> Self {
        Linear {
            weights: Array2::from_elem((input_shape, output_shape), 0.),
            bias: Array2::from_elem((1, output_shape), 0.),
            activation_function,
        }
    }

    pub fn new_from_rand(input_shape: usize, output_shape: usize, activation_function: ActivationFunction) -> Self {
        let mut rng = rand::thread_rng();
        Linear {
            weights: Array2::from_shape_fn((input_shape, output_shape), |(_i, _j)| rng.gen_range(-1., 1.)),
            bias: Array2::from_shape_fn((1, output_shape), |(_i, _j)| rng.gen_range(-1., 1.)),
            activation_function,
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        Dot::dot(input, &self.weights) + &self.bias
        // input.dot(&self.weights) + &self.bias
        // self.weights.dot(input) + &self.bias
    }

    pub fn activate(&self, activations: Array2<f32>) -> Array2<f32> {
        self.activation_function.plain(activations)
    }

    pub fn backward(&self, 
        activations: &Array2<f32>,
        prev_activations: &Array2<f32>,
        error: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>) {

        // Compute gradients for weights and biases
        
        let predicted_activations = error * self.activation_function.derivative(activations.clone());
        let shape = 1.;// prev_activations.shape()[1] as f32;

        let weight_gradient = prev_activations.t().dot(&predicted_activations) / shape;
        let bias_gradient_vec = (predicted_activations.sum_axis(Axis(0)) / shape).to_vec();
        let bias_gradient = Array2::from_shape_vec((1, bias_gradient_vec.len()), bias_gradient_vec).unwrap();

        /* println!("weight gradient {}", weight_gradient);
        println!("bias gradient {}", bias_gradient); */

        // Compute the input gradient to propagate backward
        let next_input = predicted_activations.dot(&self.weights.t()); // self.weights.t().dot(&activated_deltas);
        (next_input, weight_gradient, bias_gradient)
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
