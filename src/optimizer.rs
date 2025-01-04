use ndarray::Array2;

use crate::neural_network::NeuralNetwork;

pub trait Optimizer {
    fn update(&mut self, network: &mut NeuralNetwork, gradients: &[(Array2<f32>, Array2<f32>)]);
}
