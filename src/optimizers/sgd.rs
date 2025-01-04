use ndarray::Array2;

use crate::{neural_network::NeuralNetwork, optimizer::Optimizer};

pub struct SGD {
    pub learning_rate: f32,
}

impl Optimizer for SGD {
    fn update(&mut self, network: &mut NeuralNetwork, gradients: &[(Array2<f32>, Array2<f32>)]) {
        ()
    }
}
