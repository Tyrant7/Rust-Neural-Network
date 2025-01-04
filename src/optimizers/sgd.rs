use crate::optimizer::Optimizer;

pub struct SGD {
    pub learning_rate: f32,
}

impl Optimizer for SGD {
    fn update(&mut self, params: &mut [ndarray::Array2<f32>], grads: &[ndarray::Array2<f32>]) {
        ()
    }
}
