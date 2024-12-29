use crate::layer::Layer;

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl Layer for ReLU {
    fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        input.into_iter().map(|x| x.max(0.)).collect()
    }

    fn backward(&self, activations: Vec<f32>) -> Vec<f32> {
        activations.into_iter().map(|x| if x > 0. { 1. } else { 0. }).collect()
    }
}
