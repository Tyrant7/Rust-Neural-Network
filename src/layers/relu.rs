use crate::layer::{Layer, LayerShape};

pub struct ReLU {
    shape: LayerShape,
}

impl Layer for ReLU {
    fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        input.into_iter().map(|x| x.max(0.)).collect()
    }

    fn backward(&self) -> Vec<f32> {
        
    }
}
