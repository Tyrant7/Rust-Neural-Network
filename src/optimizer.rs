use ndarray::Array2;

use crate::neural_network::NeuralNetwork;

pub trait Optimizer {
    fn update(&mut self, network: &mut NeuralNetwork, weight_gradients: &[Array2<f32>], bias_gradients: &[Array2<f32>]);
}
