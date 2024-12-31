use ndarray::{Array2, Axis, Shape};

use crate::layer::Layer;

pub struct Linear {
    weights: Array2<f32>,
    bias: Array2<f32>,
}

impl Linear {
    pub fn new(input_shape: usize, output_shape: usize) -> Self {
        Linear {
            weights: Array2::from_elem((output_shape, input_shape), 0.),
            bias: Array2::from_elem((input_shape, 1), 0.),
        }
    }

    pub fn forward(&self, input: Array2<f32>) -> Array2<f32> {
        self.weights.dot(&input) + &self.bias
    }

    pub fn backward(&self, activations: Array2<f32>) -> Array2<f32> {
        activations.dot(&self.weights.t())
    }

    pub fn compute_bias_gradient(&self, activations: Array2<f32>) -> Array2<f32> {
        activations.sum_axis(Axis(1)).insert_axis(Axis(1))
    }
}

impl Layer {
    pub fn linear(input_size: usize, output_size: usize) -> Self {
        Layer::Linear(Linear::new(input_size, output_size))
    }
}
