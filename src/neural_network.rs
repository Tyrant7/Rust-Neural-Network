use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};
use std::cmp::max;

use rand::Rng;
extern crate rand;

pub struct NeuralNetworkManager {
    id_index: i32,
    pub networks: Vec<( i32, NeuralNetwork )>,
}

impl NeuralNetworkManager {
    pub fn new(&mut self) {

        return;
    }
    pub fn new_id(&mut self) -> String {
        
        self.id_index += 1;
        return (&self.id_index).to_string()
    }
}

pub static NEURAL_NETWORK_MANAGER: Mutex<NeuralNetworkManager> = Mutex::new(NeuralNetworkManager {
    id_index: 1,
    networks: vec![],
})/* .unwrap() */;

const BIAS: usize = 1;
const LEARNING_RATE: usize = 1;
const HIDDEN_LAYERS_COUNT: usize = 2;
const HIDDEN_PERCEPTRON_COUNT: usize = 3;


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

#[derive(Clone)]
pub struct Input {
    pub name: String,
    pub value: usize,
    pub weightID: String,
}

pub struct Output {
    pub name: String,
}

pub struct NeuralNetwork {
    pub id: String,
    pub weight_layers: Vec<Vec<Vec<usize>>>,
    /**
     * An ID reference to weights for a set of input perceptrons
     */
    pub input_weights: HashMap<String, usize>,
    /**
     * An array of IDs to find the input's weight
     */
    pub input_weight_layers: Vec<String>,
    pub activation_layers: Vec<Vec<usize>>,
}

impl NeuralNetwork {
    pub fn new(mut self/* , weight_layers: Option<Vec<Vec<Vec<usize>>>>, activation_layers: Option<Vec<Vec<usize>>> */) {

        

        /* self.id = NeuralNetworkManager::new_id(NeuralNetworkManager); */

        /* if let Some(self.weight_layers) { self.weight_layers = weight_layers }; */
/* 
        if let Some(weight_layers) = weight_layers {
            self.weight_layers = weight_layers;
        };

        if let Some(weight_layers) = weight_layers {
            self.weight_layers = weight_layers;
        };

        if let Some(activation_layers) = activation_layers {
            self.activation_layers = activation_layers
        }
         */
    }

    pub fn build(&mut self, inputs: &Vec<Input>, output_count: usize) {

        self.weight_layers.push(vec![]);
        self.activation_layers.push(vec![]);

        // Construct the input layer

        let mut input_i = 0;
        while input_i < inputs.len() {
            println!("{:?}", self.input_weight_layers);
            self.input_weight_layers[input_i as usize] = inputs[input_i].weightID.clone();
            self.input_weights.insert(inputs[input_i].weightID.clone(), 0);
            
            self.activation_layers[input_i as usize].push(input_i as usize);

            input_i += 1;
        }

        // Construct hidden layers

        let mut layer_i = 1;
        while layer_i < HIDDEN_LAYERS_COUNT {

            self.weight_layers.push(vec![]);
            self.activation_layers.push(vec![]);

            let mut perceptron_i = 0;
            while perceptron_i < HIDDEN_PERCEPTRON_COUNT {

                self.weight_layers[layer_i as usize].push(vec![]);

                let mut activation_i = 0;
                while activation_i < self.activation_layers[(layer_i - 1) as usize].len() {

                    self.weight_layers[layer_i as usize][perceptron_i as usize].push(0);

                    activation_i += 1;
                }

                self.activation_layers[layer_i as usize].push(0);

                perceptron_i += 1;
            }

            layer_i += 1;
        }

        // Output layers

        self.weight_layers.push(vec![]);
        self.activation_layers.push(vec![]);

        let last_layer_index = self.activation_layers.len() - 1;

        let mut output_i = 0;
        while output_i < output_count {

            self.weight_layers[last_layer_index].push(vec![]);

            let mut activation_i = 0;
            while activation_i < self.activation_layers[last_layer_index - 1].len() {

                self.weight_layers[last_layer_index][output_i].push(0);

                activation_i += 1;
            }

            self.activation_layers[last_layer_index].push(0);

            output_i += 1;
        }
    }

    /**
     * 
     */
    pub fn forward_propagate(&mut self, inputs: &Vec<Input>) {

        let mut input_i = 0;
        while input_i < inputs.len() {

            self.activation_layers[0][input_i] = max(0, inputs[input_i].value * self.input_weights[&inputs[input_i].weightID] + BIAS);
            inputs[input_i].value;
            
            input_i += 1;
        }
        
        let mut layer_i = 1;
        while layer_i < self.activation_layers.len() {

            let mut activation_index = 0;
            while activation_index < self.activation_layers[layer_i].len() {

                self.activation_layers[layer_i][activation_index] = 0;
                
                let mut previous_layer_i = 0;
                while previous_layer_i < self.activation_layers[(layer_i - 1) as usize].len() {

                    self.activation_layers[layer_i][activation_index] += self.activation_layers[layer_i][previous_layer_i] * self.weight_layers[layer_i][activation_index][previous_layer_i];

                    previous_layer_i += 1;
                }
                
                self.activation_layers[layer_i][activation_index] = max(0, self.activation_layers[layer_i][activation_index] + BIAS);

                activation_index += 1;
            }

            layer_i += 1;
        }
    }

    pub fn back_propagate(&mut self, scored_outputs: bool) {


    }

    /**
     * Randomly increases or decreases weights
     */
    pub fn mutate(&mut self, inputs: &Vec<Input>) {

        let mut rng = rand::thread_rng();

        // Input layer

        let mut input_i = 0;
        while input_i < self.input_weights.len() {

            if input_i >= inputs.len() {

                break;
            };

            let new_weight = self.input_weights[&inputs[input_i].weightID] + rng.gen::<usize>() * LEARNING_RATE - rng.gen::<usize>() * LEARNING_RATE;
            self.input_weights.insert(inputs[input_i].weightID.clone(), new_weight);
            
            input_i += 1;
        }

        // Other layers

        let mut layer_i = 1;
        while layer_i < self.activation_layers.len() {

            let mut activation_index = 0;
            while activation_index < self.activation_layers[layer_i].len() {
                
                let mut weight_i = 0;
                
                while weight_i < self.weight_layers[layer_i][activation_index].len() {
                 
                    self.weight_layers[layer_i][activation_index][weight_i] += rng.gen::<usize>() * LEARNING_RATE - rng.gen::<usize>() * LEARNING_RATE;
                    weight_i += 1;
                }

                activation_index += 1;
            }
            
            layer_i += 1;
        }
    }

    pub fn init_visuals(&mut self) {


    }

    pub fn update_visuals(&mut self) {


    }

    pub unsafe fn clone(&self) -> NeuralNetwork  {

        let new_neural_network = NeuralNetwork {
            id: NEURAL_NETWORK_MANAGER.lock().unwrap().new_id(),
            input_weight_layers: self.input_weight_layers.clone(),
            input_weights: self.input_weights.clone(),
            weight_layers: self.weight_layers.clone(),
            activation_layers: self.activation_layers.clone(),
        };
        /* new_neural_network.new(); */

        return new_neural_network;
    }
}
