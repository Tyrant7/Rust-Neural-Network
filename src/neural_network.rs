use std::fs;
use std::sync::Mutex;

use ndarray::prelude::*;
use rand::Rng;

use crate::utils::{relu, relu_derivative};
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

pub struct NeuralNetwork {
    pub id: u32,
    pub learning_rate: f64,
    /// A list of each layer (by index) with values describing the amount of perceptrons in the layer
    pub layers: Vec<usize>,
    pub bias_layers: Vec<Array2<f64>>,
    pub weight_layers: Vec<Array2<f64>>,
}

impl NeuralNetwork {
    //     pub fn new(mut self/* , weight_layers: Option<Vec<Vec<Vec<usize>>>>, activation_layers: Option<Vec<Vec<usize>>> */) {

    //         /* self.id = NeuralNetworkManager::new_id(NeuralNetworkManager); */
    //         /* if let Some(self.weight_layers) { self.weight_layers = weight_layers }; */
    //          /*
    //         if let Some(weight_layers) = weight_layers {
    //             self.weight_layers = weight_layers;
    //         };

    //         if let Some(weight_layers) = weight_layers {
    //             self.weight_layers = weight_layers;
    //         };

    //         if let Some(activation_layers) = activation_layers {
    //             self.activation_layers = activation_layers
    //         }
    //          */
    //     }

    pub fn new(bias: f64, learning_rate: f64, layers: Vec<usize>) -> Self {
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

    fn empty_weight_layers(layers: &[usize]) -> Vec<Array2<f64>> {
        let mut weight_layers: Vec<Array2<f64>> = Vec::new();

        // Input layers

        let mut layer_vec = Vec::new();

        for _ in 0..layers[0] {
            layer_vec.push(0.);
        }

        weight_layers.push(Array2::from_shape_vec((1, layers[0]), layer_vec).unwrap());

        // Hidden and output layers

        for layer_i in 1..layers.len() - 1 {
            let mut layer_vec = Vec::new();

            for _ in 0..layers[layer_i] {
                for _ in 0..layers[layer_i - 1] {
                    layer_vec.push(0.);
                }
            }

            weight_layers.push(
                Array2::from_shape_vec((layers[layer_i], layers[layer_i - 1]), layer_vec).unwrap(),
            );
        }

        weight_layers
    }

    fn empty_bias_layers(layers: &[usize], bias: f64) -> Vec<Array2<f64>> {
        let mut bias_layers: Vec<Array2<f64>> = Vec::new();

        // Input layers

        let mut layer_vec = Vec::new();

        for _ in 0..layers[0] {
            layer_vec.push(bias);
        }

        bias_layers.push(Array2::from_shape_vec((1, layers[0]), layer_vec).unwrap());

        // Hidden and output layers

        for layer_i in 1..layers.len() - 1 {
            let mut layer_vec = Vec::new();

            for _ in 0..layers[layer_i] {
                for _ in 0..layers[layer_i - 1] {
                    layer_vec.push(bias);
                }
            }

            bias_layers.push(
                Array2::from_shape_vec((layers[layer_i], layers[layer_i - 1]), layer_vec).unwrap(),
            );
        }

        bias_layers
    }

    /**
     *
     */
    pub fn forward_propagate(&self, inputs: &[f64]) -> Vec<Array2<f64>> {
        #[cfg(feature = "debug_network")]
        println!("Foward prop");
        
        // Don't do weight opperations on the input layer
        // For the second layers and beyond, use dot product
        // construct input layer from inputs vec
        // 

        let mut activation_layers: Vec<Array2<f64>> = Vec::new();

        // Construct activation layers

        // Input Layers

        let mut layer_vec = Vec::new();

        for input in inputs {
            layer_vec.push(*input);
        }
        
        let array: Array2<f64> = Array2::from_shape_vec((1, self.layers[0]), layer_vec)
            .unwrap()
            * &self.weight_layers[0]
            + &self.bias_layers[0];
        activation_layers.push(array.map(|x| relu(*x)));

        // Hidden and output layers

        for layer_i in 1..self.layers.len() - 1 {
            let mut layer_vec = Vec::new();

            // for perceptron_i in 0..self.layers[layer_i] {
            //     // Looping through way too many values
            //     for previous_i in activation_layers[layer_i - 1].iter() {
            //         activation_layers[]
            //         layer_vec.push(*previous);
            //     }
            // }

            for previous in activation_layers[layer_i - 1].iter() {
                layer_vec.push(*previous);
            }

            println!("layer vec {}", layer_vec.len());
            println!("shape {} {}", self.layers[layer_i], self.layers[layer_i - 1]);
            println!("weight layers {}", self.weight_layers[layer_i].len());
            println!("bias layers {}", self.bias_layers[layer_i].len());

            for weights in self.weight_layers[layer_i].iter() {
                
            }

            let x = &activation_layers[layer_i - 1] * &self.weight_layers[layer_i];
            println!("x {}", x);

            // for each perceptron's weights, 


            let array =
                Array2::from_shape_vec((self.layers[layer_i - 1], self.layers[layer_i]), layer_vec)
                    .unwrap()
                    * &self.weight_layers[layer_i]
                    + &self.bias_layers[layer_i];
            activation_layers.push(array.map(|x| relu(*x)));
        }

        #[cfg(feature = "debug_network")]
        println!("{:?}", activation_layers);

        activation_layers
    }

    pub fn backwards_propagate(&mut self, activation_layers: &[Array2<f64>]) -> Vec<Array2<f64>> {
        let mut gradients = Vec::new();

        // Output layer



        // Middle and input layer

        for layer_i in (0..self.layers.len() - 2).rev() {
            // Previous times current, then relu derivatived
            let activations = &gradients[layer_i - 1] * &activation_layers[layer_i];
            gradients.push(activations.map(|x| relu_derivative(*x)));   
        }

        gradients
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

impl Clone for NeuralNetwork {
    fn clone(&self) -> NeuralNetwork {
        NeuralNetwork {
            learning_rate: self.learning_rate,
            layers: self.layers.clone(),
            id: NEURAL_NETWORK_MANAGER.lock().unwrap().new_id(),
            bias_layers: self.bias_layers.clone(),
            weight_layers: self.weight_layers.clone(),
        }
    }
}
