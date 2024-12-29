use crate::layer::{Layer, LayerShape};

pub struct ReLU {
    shape: LayerShape,
}

impl ReLU {
    pub fn new(input_shape: u32, output_shape: u32) -> Self {
        ReLU {
            shape: LayerShape::new(input_shape, output_shape),
        }
    }
}

impl Layer for ReLU {
    fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        input.into_iter().map(|x| x.max(0.)).collect()
    }

    fn backward(&self) -> Vec<f32> {
        vec![] // TODO
    }
}
