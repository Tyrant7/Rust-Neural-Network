use ndarray::{Array2, Shape};

use crate::layer::{Layer, LayerShape};

pub struct Linear {
    shape: LayerShape,
    weights: Array2<f32>,
    bias: Array2<f32>,
}

impl Linear {
    pub fn new(input_shape: usize, output_shape: usize) -> Self {
        Linear {
            shape: LayerShape::new(input_shape, output_shape),
            weights: Array2::from_elem((output_shape, input_shape), 0.),
            bias: Array2::from_elem((input_shape, 1), 0.),
        }
    }
}

impl Layer for Linear {
    fn forward(&self, input: Array2<f32>) -> Array2<f32> {
        input // TODO
    }

    fn backward(&self, activations: Array2<f32>) -> Array2<f32> {
        Array2::from_elem((0, 0), 0.) // TODO
    }
}
