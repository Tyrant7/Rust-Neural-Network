use ndarray::Array2;

use crate::layers::{
    activation_functions::activation_functions::{relu, relu_derivative, sigmoid, sigmoid_derivative}, 
    linear::Linear
};

pub enum Layer {
    ReLU,
    Sigmoid,
    Linear(Linear),
}

impl Layer {
    pub fn forward(&self, input: Array2<f32>) -> Array2<f32> {
        match self {
            Layer::ReLU => relu(input),
            Layer::Sigmoid => sigmoid(input),
            Layer::Linear(layer) => layer.forward(input),
        }
    }

    pub fn backward(&self, activations: Array2<f32>) -> Array2<f32> {
        match self {
            Layer::ReLU => relu_derivative(activations),
            Layer::Sigmoid => sigmoid_derivative(activations),
            Layer::Linear(layer) => layer.backward(activations),
        }
    }

    pub fn compute_bias_gradient(&self, activations: Array2<f32>) -> Array2<f32> {
        match self {
            Layer::Linear(layer) => layer.compute_bias_gradient(activations),
            // Some layers like activation functions won't have biases
            _ => activations
        }
    }
}
