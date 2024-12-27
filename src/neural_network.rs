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
    pub learning_rate: f32,
    /// A list of each layer (by index) with values describing the amount of perceptrons in the layer
    pub layers: Vec<usize>,
    pub bias_layers: Vec<Array2<f32>>,
    pub weight_layers: Vec<Array2<f32>>,
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
     *
     */
    pub fn forward_propagate(&self, inputs: Vec<f32>) -> Vec<Array2<f32>> {
        #[cfg(feature = "debug_network")]
        println!("Foward prop");

        // Don't do weight opperations on the input layer
        // For the second layers and beyond, use dot product
        // construct input layer from inputs vec
        //

        let mut activation_layers: Vec<Array2<f32>> = Vec::new();
        let inputs_array: Array2<f32> = Array2::from_shape_vec((self.layers[0], 1), inputs).unwrap();

        // Hidden and output layers

        for layer_i in 0..self.layers.len() - 1 {
            // let mut layer_vec = Vec::new();
            // println!("doing layer {}", layer_i);
            // for perceptron_i in 0..self.layers[layer_i] {
            //     // Looping through way too many values
            //     for previous_i in activation_layers[layer_i - 1].iter() {
            //         activation_layers[]
            //         layer_vec.push(*previous);
            //     }
            // }

            // for previous in activation_layers[layer_i - 1].iter() {
            //     layer_vec.push(*previous);
            // }

            // println!("layer vec {}", layer_vec.len());
            // println!(
            //     "shape {}",
            //     self.layers[layer_i]
            // );
            // println!("weight layers {}", self.weight_layers[layer_i]);
            // println!("bias layers {}", self.bias_layers[layer_i]);

            // for weights in self.weight_layers[layer_i].iter() {

            // }

            // let activations = &activation_layers[layer_i - 1] * &self.weight_layers[layer_i]
            //     + &self.bias_layers[layer_i];
            // println!("x {}", activations);

            // for each perceptron's weights,

            // let transfers =
            //     Array2::from_shape_vec((self.layers[layer_i - 1], self.layers[layer_i]), layer_vec)
            //         .unwrap()
            //         * &self.weight_layers[layer_i]
            //         + &self.bias_layers[layer_i];

            let previous_activations: &Array2<f32> = activation_layers.get(layer_i.saturating_sub(1)).unwrap_or(&inputs_array);
            // println!("previous_activations {}", previous_activations);
            let transfers = self.weight_layers[layer_i].dot(
                previous_activations,
            ) + &self.bias_layers[layer_i];
            
            let activations = transfers.mapv(relu);
            activation_layers.push(activations);
        }

        #[cfg(feature = "debug_network")]
        println!("{:?}", activation_layers);

        activation_layers
    }

    pub fn backwards_propagate(&mut self, activation_layers: &[Array2<f32>]) -> Vec<Array2<f32>> {

        // Not done yet and needs changes

        let mut gradient_layers = Vec::new();

        // Output layer

        // Middle and input layer

        for layer_i in (0..self.layers.len() - 1).rev() {
            // Previous times current, then relu derivatived
            let previous_activations = activation_layers[layer_i].clone();
            let activations: Array2<f32> = &gradient_layers[layer_i - 1] * &activation_layers[layer_i];

            let gradients: Array2<f32> = array![[0.]];

            let da: Array2<f32> = &gradient_layers[layer_i - 1] * activations.mapv(relu_derivative);
            let dw = (1.0 / &activation_layers[layer_i]) * (da.dot(&previous_activations.reversed_axes()));
            let db = &self.bias_layers[layer_i];
            let da_prev = self.weight_layers[layer_i].dot(&da);

            // Update weights

            self.weight_layers[layer_i] = self.weight_layers[layer_i].clone() - (gradients * self.learning_rate);

            // Update biases

            //

            gradient_layers.push(activations);
        }

        gradient_layers
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
