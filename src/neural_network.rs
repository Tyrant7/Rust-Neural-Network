use std::borrow::Borrow;
use std::collections::HashMap;
use std::ops::Index;
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
const HIDDEN_LAYERS_COUNT: usize = 1;
const HIDDEN_PERCEPTRON_COUNT: usize = 2;


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
    pub values: Vec<usize>,
    pub weight_id: String,
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
        println!("Build");
        self.weight_layers.push(vec![]);
        self.activation_layers.push(vec![]);

        // Construct the input layer

        let mut input_i = 0;
        while input_i < inputs.len() {

            self.input_weight_layers.push(inputs[input_i].weight_id.clone());
            self.input_weights.insert(inputs[input_i].weight_id.clone(), BIAS);
            
            self.weight_layers[0].push(vec![]);
            self.activation_layers[0].push(0);

            let input = &inputs[input_i];
    
            let mut value_i = 0;
            while value_i < input.values.len() {

                self.weight_layers[0][input_i].push(BIAS);
                value_i += 1;
            }
    
            input_i += 1;
        }

        // Construct hidden layers

        let mut layer_i = 1;
        while layer_i < HIDDEN_LAYERS_COUNT + 1 {

            self.weight_layers.push(vec![]);
            self.activation_layers.push(vec![]);

            let mut perceptron_i = 0;
            while perceptron_i < HIDDEN_PERCEPTRON_COUNT {

                self.weight_layers[layer_i as usize].push(vec![]);

                let mut activation_i = 0;
                while activation_i < self.activation_layers[(layer_i - 1) as usize].len() {

                    self.weight_layers[layer_i as usize][perceptron_i as usize].push(BIAS);

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

                self.weight_layers[last_layer_index][output_i].push(BIAS);

                activation_i += 1;
            }

            self.activation_layers[last_layer_index].push(0);

            output_i += 1;
        }

        println!("{:?}", self.activation_layers);
    }

    /**
     * 
     */
    pub fn forward_propagate(&mut self, inputs: &Vec<Input>) {
        println!("Foward prop");

        let mut activation_i = 0;
        while activation_i < self.activation_layers[0].len() {
            
            self.activation_layers[0][activation_i] = 0;
            activation_i += 1;
        }
        
        let mut input_i = 0;
        while input_i < inputs.len() {

            let input = &inputs[input_i];

            let mut value_i = 0;
            while value_i < input.values.len() {

                self.activation_layers[0][input_i] += max(0, inputs[input_i].values[value_i] * self.input_weights[&inputs[input_i].weight_id]);
                value_i += 1;
            }

            input_i += 1;
        }
        println!("{:?}", self.activation_layers);
        println!("A");
        //
        
        let mut layer_i = 1;
        while layer_i < self.activation_layers.len() {

            activation_i = 0;
            while activation_i < self.activation_layers[layer_i].len() {

                self.activation_layers[layer_i][activation_i] = 0;
                
                let mut previous_layer_activation_i = 0;
                while previous_layer_activation_i < self.activation_layers[(layer_i - 1) as usize].len() {
                    println!("{:?}", self.activation_layers);
                    println!("{}", layer_i);
                    println!("{}", activation_i);
                    println!("{}", previous_layer_activation_i);
                    self.activation_layers[layer_i][activation_i] += self.activation_layers[layer_i - 1][previous_layer_activation_i] * self.weight_layers[layer_i][activation_i][previous_layer_activation_i];

                    previous_layer_activation_i += 1;
                }
                
                self.activation_layers[layer_i][activation_i] = max(0, self.activation_layers[layer_i][activation_i]);

                activation_i += 1;
            }

            layer_i += 1;
        }
        print!("End of forward prop");
        println!("{:?}", self.activation_layers);
    }

    pub fn back_propagate(&mut self, scored_outputs: bool) {


    }

    /**
     * Randomly increases or decreases weights
     */
    pub fn mutate(&mut self) {

        let mut rng = rand::thread_rng();

        // Input layer

        let mut input_i = 0;
        while input_i < self.input_weight_layers.len() {

            let weight_id = self.input_weight_layers[input_i].to_string();
            let present_weight = self.input_weights.get(&weight_id).unwrap();
            let new_weight = present_weight + rng.gen::<usize>() * LEARNING_RATE - rng.gen::<usize>() * LEARNING_RATE;

            self.input_weights.insert(weight_id, new_weight);

            input_i += 1;
        }

            /* 
            let mut weight_i = 0;
                
            while weight_i < self.weight_layers[0][input_i].len() {
             
                self.weight_layers[0][input_i][weight_i] += rng.gen::<usize>() * LEARNING_RATE - rng.gen::<usize>() * LEARNING_RATE;
                weight_i += 1;
            }
             */

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
