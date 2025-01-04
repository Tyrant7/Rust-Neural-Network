use ndarray::Array2;

pub trait Optimizer {
    fn update(&mut self, params: &mut [Array2<f32>], grads: &[Array2<f32>]);
}
