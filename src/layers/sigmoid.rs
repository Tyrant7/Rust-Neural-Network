use ndarray::Array2;

use crate::layer::Layer;

pub struct Sigmoid;

impl Sigmoid {
    pub fn forward(&self, input: Array2<f32>) -> Array2<f32> {
        input.mapv(sigmoid)
    }

    pub fn backward(&self, activations: Array2<f32>) -> Array2<f32> {
        activations.mapv(|x| sigmoid(x) * (1. - sigmoid(x)))
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
