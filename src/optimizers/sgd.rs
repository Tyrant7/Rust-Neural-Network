use ndarray::Array2;

use crate::{neural_network::NeuralNetwork, optimizer::Optimizer};

pub struct SGD {
    pub learning_rate: f32,
}

impl Optimizer for SGD {
    fn update(&mut self, network: &mut NeuralNetwork, weight_gradients: &[Array2<f32>], bias_gradients: &[Array2<f32>]) {
        for ((layer, weight_gradients), bias_gradients) in network.layers.iter_mut().zip(weight_gradients).zip(bias_gradients) {

            // println!("[opt] bias grads: {}", bias_gradients);
            // println!("[opt] biases {}", layer.get_params_mut().1);

            println!("[opt] weight grads: {}", weight_gradients);
            println!("[opt] weights {}", layer.get_params_mut().0);

            let (weights, biases) = layer.get_params_mut();
            for (weight, weight_grad) in weights.iter_mut().zip(weight_gradients) {
                // *weight += weight_grad * self.learning_rate;
                *weight += (weight_grad * self.learning_rate) * 2.;
            }
            for (bias, bias_grad) in biases.iter_mut().zip(bias_gradients) {
                *bias += bias_grad * self.learning_rate;
            }
        }
    }
}
