pub trait Layer {
    fn forward(&self, input: Vec<f32>) -> Vec<f32>;
    fn backward(&self) -> Vec<f32>;
}

pub struct LayerShape {
    pub input_shape: u32,
    pub output_shape: u32,
}

impl LayerShape {
    pub fn new(input_shape: u32, output_shape: u32) -> Self {
        LayerShape {
            input_shape,
            output_shape,
        }
    }

    pub fn input_shape(&self) -> u32 {
        self.input_shape
    }

    pub fn output_shape(&self) -> u32 {
        self.output_shape
    }
}
