use ndarray::Array2;

use crate::layers::{
    activation_functions::{relu, relu_derivative, sigmoid, sigmoid_derivative}, 
    linear::Linear
};

pub type ActivationFunction = fn(input: &Array2<f32>) -> Array2<f32>;

pub enum Layer {
    Linear(Linear),
}

impl Layer {
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            Layer::Linear(layer) => layer.forward(input),
        }
    }

    pub fn backward(&self, activations: &Array2<f32>) -> Array2<f32> {
        match self {
            Layer::Linear(layer) => layer.backward(activations),
        }
    }
}
