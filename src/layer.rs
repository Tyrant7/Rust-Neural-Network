use ndarray::Array2;

use crate::layers::linear::Linear;

#[derive(Debug)]
pub enum Layer {
    Linear(Linear),
}

impl Layer {
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            Layer::Linear(layer) => layer.forward(input),
        }
    }

    pub fn activate(&self, transfers: Array2<f32>) -> Array2<f32> {
        match self {
            Layer::Linear(layer) => layer.activate(transfers),
        }
    }

    pub fn backward(&self,    
        activations: &Array2<f32>,  
        prev_activations: &Array2<f32>,
        error: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        match self {
            Layer::Linear(layer) => layer.backward(activations, prev_activations, error),
        }
    }

    pub fn get_params(&self) -> (&Array2<f32>, &Array2<f32>) {
        match self {
            Layer::Linear(layer) => layer.get_params(),
        }
    }

    pub fn get_params_mut(&mut self) -> (&mut Array2<f32>, &mut Array2<f32>) {
        match self {
            Layer::Linear(layer) => layer.get_params_mut(),
        }
    }
}
