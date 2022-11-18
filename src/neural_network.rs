use std::sync::{Mutex, MutexGuard};
use std::cmp::max;
extern crate rand;

pub struct NeuralNetworkManager {
    id_index: i32,
    networks: Vec<( i32, NeuralNetwork )>,
    bias: f32,
    learning_rate: f32,
    hidden_layers_count: usize,
    hidden_perceptron_count: usize,
}

impl NeuralNetworkManager {
    pub fn init(&mut self) {

        return;
    }
    pub fn new_id(&mut self) -> i32 {
        
        self.id_index += 1;
        return self.id_index
    }
}

static neural_network_manager: MutexGuard<NeuralNetworkManager> = Mutex::new(NeuralNetworkManager {
    id_index: 1,
    networks: vec![],
    bias: 1.,
    learning_rate: 1.,
    hidden_layers_count: 2,
    hidden_perceptron_count: 3,
}).lock().unwrap();

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
    id: i32,
    weight_layers: Vec<Vec<Vec<f32>>>,
    activation_layers: Vec<Vec<f32>>,
}

struct Input {
    name: String,
    value: f32
}

struct Output {
    name: String,
}

impl NeuralNetwork {
    pub fn init(&mut self, weight_layers: Option<Vec<Vec<Vec<f32>>>>, activation_layers: Option<Vec<Vec<f32>>>) {
        /* self.id = NeuralNetworkManager::new_id(NeuralNetworkManager); */

        /* if let Some(self.weight_layers) { self.weight_layers = weight_layers }; */
        if let Some(weight_layers) = weight_layers {
            self.weight_layers = weight_layers;
        };

        if let Some(activation_layers) = activation_layers {
            self.activation_layers = activation_layers
        }

        return
    }

    fn build(&mut self, input_count: usize, output_count: usize) {

        self.weight_layers.push(vec![]);
        self.activation_layers.push(vec![]);

        // Construct the input layer

        let mut input_i = 0;
        while input_i < input_count {

            self.weight_layers[input_i as usize].push(vec![input_i as f32]);
            self.activation_layers[input_i as usize].push(input_i as f32);

            input_i += 1;
        }

        // Construct hidden layers

        let mut layer_i = 0;
        while layer_i < neural_network_manager.hidden_layers_count {

            self.weight_layers.push(vec![]);
            self.activation_layers.push(vec![]);

            let mut perceptron_i = 0;
            while perceptron_i < neural_network_manager.hidden_perceptron_count {

                self.weight_layers[layer_i as usize].push(vec![]);

                let mut activation_i = 0;
                while activation_i < self.activation_layers[(layer_i - 1) as usize].len() {

                    self.weight_layers[layer_i as usize][perceptron_i as usize].push(0.);

                    activation_i += 1;
                }

                self.activation_layers[layer_i as usize].push(0.);

                perceptron_i += 1;
            }

            layer_i += 1;
        }

        self.weight_layers.push(vec![]);
        self.activation_layers.push(vec![]);

        let last_layer_index = self.activation_layers.len() - 1;

        input_i = 0;
        while input_i < output_count {

            self.weight_layers[last_layer_index].push(vec![]);

            let mut activation_i = 0;
            while activation_i < self.activation_layers[last_layer_index - 1].len() {

                self.weight_layers[last_layer_index][input_i].push(0.);

                activation_i += 1;
            }

            self.activation_layers[last_layer_index].push(0.);

            input_i += 1;
        }
    }

    fn forward_propagate(&mut self, inputs: Vec<Input>) {

        let mut input_i = 0;
        while input_i < inputs.len() {

            self.activation_layers[0][input_i] = max(0., inputs[input_i].value * self.weight_layers[0][input_i] + neural_network_manager.bias);
            inputs[input_i].value;
            
            input_i += 1;
        }
        
        let mut layer_i = 1;
        while layer_i < self.activation_layers.len() {

            let mut activation_index = 0;
            while activation_index < self.activation_layers[layer_i].len() {

                self.activation_layers[layer_i][activation_index] = 0.;
                
                let mut previous_layer_i = 0;
                while previous_layer_i < self.activation_layers[(layer_i - 1) as usize].len() {

                    self.activation_layers[layer_i][activation_index] += self.activation_layers[layer_i][previous_layer_i] * self.weight_layers[layer_i][activation_index][previous_layer_i];

                    previous_layer_i += 1;
                }
                
                self.activation_layers[layer_i][activation_index] = max(0., self.activation_layers[layer_i][activation_index] + neural_network_manager.bias);

                activation_index += 1;
            }

            layer_i += 1;
        }
    }

    fn back_propagate(&mut self, scored_outputs: bool) {


    }

    fn mutate(&mut self) {

        let mut rng = rand::thread_rng();

        let mut layer_i = 0;
        while layer_i < self.activation_layers.len() {

            let mut activation_index = 0;
            while activation_index < self.activation_layers[layer_i].len() {
                
                let mut weight_i = 0;
                
                while weight_i < self.weight_layers[layer_i][activation_index].len() {
                 
                    self.weight_layers[layer_i][activation_index][weight_i] += rng.gen::<f32>() * neural_network_manager.learning_rate - rng.gen::<f32>() * neural_network_manager.learning_rate;
                    weight_i += 1;
                }

                activation_index += 1;
            }
            
            layer_i += 1;
        }
    }

    fn init_visuals(&mut self) {


    }

    fn visualize(&mut self) {


    }

    fn clone(&self) -> NeuralNetwork {

        return NeuralNetwork {
            id: neural_network_manager.new_id(),
            weight_layers: self.weight_layers.to_vec(),
            activation_layers: self.activation_layers.to_vec(),
        };
    }
}
