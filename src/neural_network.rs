use std::collections::HashMap;
use std::fs;
use std::sync::Mutex;

use rand::Rng;
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

#[derive(Copy, Clone)]
pub enum InputName {
    X,
    Y,
}

#[derive(Copy, Clone)]
pub enum OutputName {
    Result,
}

#[derive(Clone)]
pub struct Input {
    pub name: InputName,
    pub values: Vec<f64>,
    pub weight_ids: Vec<u32>,
}

impl Input {
    pub fn new(name: InputName, values: Vec<f64>, weight_ids: Vec<u32>) -> Self {
        Input {
            name,
            values,
            weight_ids,
        }
    }
}

pub struct Output {
    pub name: OutputName,
}

impl Output {
    pub fn new(name: OutputName) -> Self {
        Output { name }
    }
}

pub struct NeuralNetwork {
    pub id: u32,
    pub bias: f64,
    pub learning_rate: f64,
    pub hidden_layers_count: u32,
    pub hidden_perceptron_count: u32,
    pub weight_layers: Vec<Vec<Vec<f64>>>,
    /**
     * An ID reference to weights for a set of input perceptrons
     */
    pub weights_by_id: HashMap<u32, f64>,
    /**
     * An input perceptron by input value weight of ids to find the input's weight
     */
    pub input_weight_layers: Vec<Vec<u32>>,
    pub activation_layers: Vec<Vec<f64>>,
}

impl Default for NeuralNetwork {
    fn default() -> Self {
        Self {
            // config
            bias: 0.,
            learning_rate: 0.1,
            hidden_layers_count: 2,
            hidden_perceptron_count: 10,
            //
            id: NEURAL_NETWORK_MANAGER.lock().unwrap().new_id(),
            input_weight_layers: vec![],
            weights_by_id: HashMap::new(),
            weight_layers: vec![],
            activation_layers: vec![],
        }
    }
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

    pub fn new(
        bias: f64,
        learning_rate: f64,
        hidden_layers_count: u32,
        hidden_perceptron_count: u32,
    ) -> Self {
        Self {
            bias,
            learning_rate,
            hidden_layers_count,
            hidden_perceptron_count,
            ..Default::default()
        }
    }

    pub fn build(&mut self, inputs: &[Input], output_count: usize) -> &mut Self {
        #[cfg(feature = "debug_network")]
        println!("Build");

        self.weight_layers.push(vec![]);
        self.activation_layers.push(vec![]);

        // Construct the input layer

        for input_i in 0..inputs.len() {
            self.input_weight_layers
                .push(inputs[input_i].weight_ids.clone());

            self.weight_layers[0].push(vec![]);
            self.activation_layers[0].push(0.);

            let input = &inputs[input_i];

            for value_i in 0..input.values.len() {
                self.weights_by_id
                    .insert(inputs[input_i].weight_ids[value_i], self.bias);
                self.weight_layers[0][input_i].push(self.bias);
            }
        }

        // Construct hidden layers

        for layer_i in 1..self.hidden_layers_count + 1 {
            self.weight_layers.push(vec![]);
            self.activation_layers.push(vec![]);

            for perceptron_i in 0..self.hidden_layers_count {
                self.weight_layers[layer_i as usize].push(vec![]);

                for _ in 0..self.activation_layers[(layer_i - 1) as usize].len() {
                    self.weight_layers[layer_i as usize][perceptron_i as usize].push(self.bias);
                }

                self.activation_layers[layer_i as usize].push(0.);
            }
        }

        // Output layers

        self.weight_layers.push(vec![]);
        self.activation_layers.push(vec![]);

        let last_layer_index = self.activation_layers.len() - 1;

        for output_i in 0..output_count {
            self.weight_layers[last_layer_index].push(vec![]);

            for _ in 0..self.activation_layers[last_layer_index - 1].len() {
                self.weight_layers[last_layer_index][output_i].push(self.bias);
            }

            self.activation_layers[last_layer_index].push(0.);
        }

        #[cfg(feature = "debug_network")]
        println!("{:?}", self.activation_layers);

        self
    }

    /**
     *
     */
    pub fn forward_propagate(&mut self, inputs: &[Input]) {
        #[cfg(feature = "debug_network")]
        println!("Foward prop");

        // Input layers

        for activation_i in 0..self.activation_layers[0].len() {
            self.activation_layers[0][activation_i] = 0.;
        }

        for (input_i, input) in inputs.iter().enumerate() {
            for value_i in 0..input.values.len() {
                self.activation_layers[0][input_i] += self.relu(
                    inputs[input_i].values[value_i]
                        * self.weights_by_id[&inputs[input_i].weight_ids[value_i]],
                );
            }
        }

        // Other layers

        for layer_i in 1..self.activation_layers.len() {
            for activation_i in 0..self.activation_layers[layer_i].len() {
                self.activation_layers[layer_i][activation_i] = 0.;

                for previous_layer_activation_i in 0..self.activation_layers[(layer_i - 1)].len() {
                    self.activation_layers[layer_i][activation_i] += self.activation_layers
                        [layer_i - 1][previous_layer_activation_i]
                        * self.weight_layers[layer_i][activation_i][previous_layer_activation_i];
                }

                self.activation_layers[layer_i][activation_i] =
                    self.relu(self.activation_layers[layer_i][activation_i]);
            }
        }

        #[cfg(feature = "debug_network")]
        println!("{:?}", self.activation_layers);
    }

    fn relu(&mut self, value: f64) -> f64 {
        value.max(0.)
    }

    pub fn back_propagate(&mut self, scored_outputs: bool) {}

    /**
     * Randomly increases or decreases weights
     */
    pub fn mutate(&mut self) {
        #[cfg(feature = "debug_network")]
        println!("Mutate");

        let mut rng = rand::thread_rng();

        // Input layer

        // Mutate weights

        // Not 100% sure this works
        for tuple in self.weights_by_id.iter_mut() {
            *tuple.1 += rng.gen_range(-self.learning_rate, self.learning_rate);
        }

        // Construct new weight layers

        for input_i in 0..self.input_weight_layers.len() {
            for value_i in 0..self.input_weight_layers[input_i].len() {
                let weight_id = self.input_weight_layers[input_i][value_i];
                let present_weight = self.weights_by_id.get(&weight_id).unwrap();

                self.weight_layers[0][input_i][value_i] = *present_weight;
            }
        }

        // Other layers

        for layer_i in 1..self.activation_layers.len() {
            for activation_index in 0..self.activation_layers[layer_i].len() {
                for weight_i in 0..self.weight_layers[layer_i][activation_index].len() {
                    self.weight_layers[layer_i][activation_index][weight_i] +=
                        rng.gen_range(-self.learning_rate, self.learning_rate);
                }
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
        fs::write("weights_by_id.txt", format!("{:?}", self.weights_by_id))
            .expect("Unable to write weights by id");

        fs::write("weight_layers.txt", format!("{:?}", self.weight_layers))
            .expect("Unable to write weight layers");
    }

    pub fn init_visuals(&mut self) {}

    pub fn update_visuals(&mut self) {}
}

impl Clone for NeuralNetwork {
    fn clone(&self) -> NeuralNetwork {
        NeuralNetwork {
            bias: self.bias,
            learning_rate: self.learning_rate,
            hidden_layers_count: self.hidden_layers_count,
            hidden_perceptron_count: self.hidden_perceptron_count,
            id: NEURAL_NETWORK_MANAGER.lock().unwrap().new_id(),
            input_weight_layers: self.input_weight_layers.clone(),
            weights_by_id: self.weights_by_id.clone(),
            weight_layers: self.weight_layers.clone(),
            activation_layers: self.activation_layers.clone(),
        }
    }
}
