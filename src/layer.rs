use ndarray::Array2;

use crate::layers::linear::Linear;

pub enum Layer {
    Linear(Linear),
}

impl Layer {
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            Layer::Linear(layer) => layer.forward(input),
        }
    }

    pub fn backward(&self,    
        activation_layers: &Array2<f32>,     
        delta: &Array2<f32>,
        weight_gradient: &mut Array2<f32>, 
        bias_gradient: &mut Array2<f32>) -> Array2<f32> {
        match self {
            Layer::Linear(layer) => layer.backward(activation_layers, delta, weight_gradient, bias_gradient),
        }
    }

    pub fn get_params_mut(&mut self) -> (&mut Array2<f32>, &mut Array2<f32>) {
        match self {
            Layer::Linear(layer) => layer.get_params_mut(),
        }
    }
}
