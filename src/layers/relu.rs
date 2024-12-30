use ndarray::Array2;

use crate::layer::Layer;

pub struct ReLU;

impl Layer for ReLU {
    fn forward(&self, input: Array2<f32>) -> Array2<f32> {
        input.mapv(|x| x.max(0.))
    }

    fn backward(&self, activations: Array2<f32>) -> Array2<f32> {
        activations.mapv(|x| if x > 0. { 1. } else { 0. })
    }
}
