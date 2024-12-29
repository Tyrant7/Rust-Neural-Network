pub trait Layer {
    fn forward(&self, input: Vec<f32>) -> Vec<f32>;
    fn backward(&self, activations: Vec<f32>) -> Vec<f32>;
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
