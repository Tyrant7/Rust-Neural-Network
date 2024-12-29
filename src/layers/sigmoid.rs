use crate::layer::Layer;

pub struct Sigmoid;

impl Layer for Sigmoid {
    fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        input.into_iter().map(|x| sigmoid(x)).collect()
    }

    fn backward(&self, activations: Vec<f32>) -> Vec<f32> {
        activations.into_iter().map(|x| sigmoid(x) * (1. - sigmoid(x))).collect()
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
