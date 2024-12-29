use crate::layer::{Layer, LayerShape};

pub struct Linear {
    shape: LayerShape,
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

impl Linear {
    pub fn new(input_shape: u32, output_shape: u32) -> Self {
        Linear {
            shape: LayerShape::new(input_shape, output_shape),
            weights: vec![vec![0.; input_shape as usize]; output_shape as usize],
            bias: vec![0.; output_shape as usize],
        }
    }
}

impl Layer for Linear {
    fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        input // TODO
    }

    fn backward(&self, activations: Vec<f32>) -> Vec<f32> {
        vec![] // TODO
    }
}
