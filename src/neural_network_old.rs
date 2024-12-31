use std::fs;
use std::sync::Mutex;

use ndarray::prelude::*;
use rand::Rng;

use crate::layers::activation_functions::{relu, relu_derivative}
extern crate rand;

pub struct NeuralNetworkManager {
    id_index: u32,
    /* pub networks: HashMap<i32, NeuralNetwork>, */
}

impl NeuralNetworkManager {
    pub fn new() -> Self {
        Self { id_index: 0 }
    }
    pub fn new_id(&mut self) -> u32 {
        self.id_index += 1;
        self.id_index
    }
}

pub static NEURAL_NETWORK_MANAGER: Mutex<NeuralNetworkManager> = Mutex::new(NeuralNetworkManager {
    id_index: 1,
})/* .unwrap() */;

/*
static neural_network_manager: Mutex<NeuralNetworkManager> = NeuralNetworkManager {
    id_index: 1,
    networks: vec![],
    hidden_layers_count: 2,
    hidden_perceptron_count: 3,
};
 */
/*
neural_network_manager: NeuralNetworkManager = NeuralNetworkManager {
    id_index: 1,
    networks: vec![],
    hidden_layers_count: 2,
    hidden_perceptron_count: 3,
};
 */

pub struct NeuralNetwork_Old {
    pub id: u32,
    pub learning_rate: f32,
    /// A list of each layer (by index) with values describing the amount of perceptrons in the layer
    pub layers: Vec<usize>,
    pub bias_layers: Vec<Array2<f32>>,
    pub weight_layers: Vec<Array2<f32>>,
}

impl NeuralNetwork_Old {
    pub fn new(bias: f32, learning_rate: f32, layers: Vec<usize>) -> Self {
        let weight_layers = Self::empty_weight_layers(&layers);
        let bias_layers = Self::empty_bias_layers(&layers, bias);

        Self {
            weight_layers,
            bias_layers,
            learning_rate,
            layers,
            id: NEURAL_NETWORK_MANAGER.lock().unwrap().new_id(),
        }
    }

    fn empty_weight_layers(layers: &[usize]) -> Vec<Array2<f32>> {
        let mut weight_layers: Vec<Array2<f32>> = Vec::new();

        // Input layers

        // Hidden and output layers

        for layer_i in 1..layers.len() {
            let mut layer_vec = Vec::new();

            // Previous layer perceptrons times current layer perceptrons
            let weights_count = (layers[layer_i]) * layers[layer_i - 1];

            for _ in 0..weights_count {
                layer_vec.push(0.);
            }

            weight_layers.push(
                Array2::from_shape_vec((layers[layer_i], layers[layer_i - 1]), layer_vec).unwrap(),
            );
        }

        weight_layers
    }

    fn empty_bias_layers(layers: &[usize], bias: f32) -> Vec<Array2<f32>> {
        let mut bias_layers: Vec<Array2<f32>> = Vec::new();

        // Hidden and output layers

        for perceptron_count in layers.iter().take(layers.len()).skip(1) {
            let mut layer_vec = Vec::new();

            for _ in 0..*perceptron_count {
                layer_vec.push(bias);
            }

            bias_layers.push(
                Array2::from_shape_vec((*perceptron_count, 1), layer_vec).unwrap(),
            );
        }

        bias_layers
    }

    /**
     * Randomly increases or decreases weights
     */
    pub fn mutate(&mut self) {
        #[cfg(feature = "debug_network")]
        println!("Mutate");

        let mut rng = rand::thread_rng();

        // Weight layers

        for weights in self.weight_layers.iter_mut() {
            for weight in weights.iter_mut() {
                *weight += rng.gen_range(-self.learning_rate, self.learning_rate);
            }
        }

        // Bias layers

        for biases in self.bias_layers.iter_mut() {
            for bias in biases.iter_mut() {
                *bias += rng.gen_range(-self.learning_rate, self.learning_rate);
            }
        }

        #[cfg(feature = "debug_network")]
        println!("{:?}", self.weight_layers);
    }

    pub fn write_to_file(&self) {
        #[cfg(feature = "debug_network")]
        println!("Write to file");

        self.write_weights();
    }

    pub fn write_weights(&self) {
        fs::write("weight_layers.txt", format!("{:?}", self.weight_layers))
            .expect("Unable to write weight layers");
    }

    pub fn write_biases(&self) {
        fs::write("bias_layers.txt", format!("{:?}", self.bias_layers))
            .expect("Unable to write bias layers");
    }

    pub fn init_visuals(&mut self) {}

    pub fn update_visuals(&mut self) {}
}

impl Clone for NeuralNetwork_Old {
    fn clone(&self) -> NeuralNetwork_Old {
        NeuralNetwork_Old {
            learning_rate: self.learning_rate,
            layers: self.layers.clone(),
            id: NEURAL_NETWORK_MANAGER.lock().unwrap().new_id(),
            bias_layers: self.bias_layers.clone(),
            weight_layers: self.weight_layers.clone(),
        }
    }
}
