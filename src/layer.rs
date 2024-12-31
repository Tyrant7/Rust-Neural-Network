use ndarray::Array2;

use crate::layers::{linear::Linear, relu::ReLU, sigmoid::Sigmoid};

pub enum Layer {
    ReLU(ReLU),
    Sigmoid(Sigmoid),
    Linear(Linear),
}

impl Layer {
    pub fn forward(&self, input: Array2<f32>) -> Array2<f32> {
        match self {
            Layer::ReLU(layer) => layer.forward(input),
            Layer::Sigmoid(layer) => layer.forward(input),
            Layer::Linear(layer) => layer.forward(input),
        }
    }

    pub fn backward(&self, activations: Array2<f32>) -> Array2<f32> {
        match self {
            Layer::ReLU(layer) => layer.backward(activations),
            Layer::Sigmoid(layer) => layer.backward(activations),
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

pub struct LayerShape {
    pub input_shape: usize,
    pub output_shape: usize,
}

impl LayerShape {
    pub fn new(input_shape: usize, output_shape: usize) -> Self {
        LayerShape {
            input_shape,
            output_shape,
        }
    }

    pub fn input_shape(&self) -> usize {
        self.input_shape
    }

    pub fn output_shape(&self) -> usize {
        self.output_shape
    }
}
