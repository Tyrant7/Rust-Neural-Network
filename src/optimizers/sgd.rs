use ndarray::Array2;

use crate::{neural_network::NeuralNetwork, optimizer::Optimizer};

pub struct SGD {
    pub learning_rate: f32,
}

impl Optimizer for SGD {
    fn update(&mut self, network: &mut NeuralNetwork, gradients: &[(Array2<f32>, Array2<f32>)]) {
        for (layer, (weight_grads, bias_grads)) in network.layers.iter_mut().zip(gradients) {

            println!("[opt] bias grads: {}", bias_grads);
            println!("[opt] biases {}", layer.get_params_mut().1);

            let (weights, biases) = layer.get_params_mut();
            for (weight, weight_grad) in weights.iter_mut().zip(weight_grads) {
                *weight -= weight_grad * self.learning_rate;
            }
            for (bias, bias_grad) in biases.iter_mut().zip(bias_grads) {
                *bias -= bias_grad * self.learning_rate;
            }
        }
    }
}
