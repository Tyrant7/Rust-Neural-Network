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
        activations: &Array2<f32>,  
        previous_transfers: &Array2<f32>,
        delta: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        match self {
            Layer::Linear(layer) => layer.backward(activations, previous_transfers, delta),
        }
    }

    pub fn get_params_mut(&mut self) -> (&mut Array2<f32>, &mut Array2<f32>) {
        match self {
            Layer::Linear(layer) => layer.get_params_mut(),
        }
    }
}
