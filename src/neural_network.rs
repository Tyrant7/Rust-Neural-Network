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

const BIAS: f64 = 1.;
const LEARNING_RATE: f64 = 1.;
const HIDDEN_LAYERS_COUNT: i32 = 1;
const HIDDEN_PERCEPTRON_COUNT: i32 = 2;

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

#[derive(Clone, Default)]
pub struct NeuralNetwork {
    pub id: u32,
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

    pub fn new() -> Self {
        /*
               self.id = NEURAL_NETWORK_MANAGER.lock().unwrap().new_id();

               NEURAL_NETWORK_MANAGER.lock().unwrap().networks.insert(self.id.parse::<i32>().unwrap(), self);
        */
        NeuralNetwork {
            id: NEURAL_NETWORK_MANAGER.lock().unwrap().new_id(),
            input_weight_layers: vec![],
            weights_by_id: HashMap::new(),
            weight_layers: vec![],
            activation_layers: vec![],
        }
    }

    pub fn build(&mut self, inputs: &[Input], output_count: usize) -> &mut Self {
        #[cfg(feature = "debug_network")]
        println!("Build");

        self.weight_layers.push(vec![]);
        self.activation_layers.push(vec![]);

        // Construct the input layer

        let mut input_i = 0;
        while input_i < inputs.len() {
            self.input_weight_layers
                .push(inputs[input_i].weight_ids.clone());

            self.weight_layers[0].push(vec![]);
            self.activation_layers[0].push(0.);

            let input = &inputs[input_i];

            let mut value_i = 0;
            while value_i < input.values.len() {
                self.weights_by_id
                    .insert(inputs[input_i].weight_ids[value_i], BIAS);
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

                self.activation_layers[layer_i as usize].push(0.);

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

            self.activation_layers[last_layer_index].push(0.);

            output_i += 1;
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

        let mut activation_i = 0;
        while activation_i < self.activation_layers[0].len() {
            self.activation_layers[0][activation_i] = 0.;
            activation_i += 1;
        }

        let mut input_i = 0;
        while input_i < inputs.len() {
            let input = &inputs[input_i];

            let mut value_i = 0;
            while value_i < input.values.len() {
                self.activation_layers[0][input_i] += self.relu(
                    inputs[input_i].values[value_i]
                        * self.weights_by_id[&inputs[input_i].weight_ids[value_i]],
                );
                value_i += 1;
            }

            input_i += 1;
        }

        //

        let mut layer_i = 1;
        while layer_i < self.activation_layers.len() {
            activation_i = 0;
            while activation_i < self.activation_layers[layer_i].len() {
                self.activation_layers[layer_i][activation_i] = 0.;

                let mut previous_layer_activation_i = 0;
                while previous_layer_activation_i
                    < self.activation_layers[(layer_i - 1)].len()
                {
                    self.activation_layers[layer_i][activation_i] += self.activation_layers
                        [layer_i - 1][previous_layer_activation_i]
                        * self.weight_layers[layer_i][activation_i][previous_layer_activation_i];

                    previous_layer_activation_i += 1;
                }

                self.activation_layers[layer_i][activation_i] =
                    self.relu(self.activation_layers[layer_i][activation_i]);

                activation_i += 1;
            }

            layer_i += 1;
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
        /*
               let mut input_i = 0;
               while input_i < self.input_weight_layers.len() {

                   let mut value_i = 0;
                   while value_i < self.input_weight_layers[input_i].len() {

                       let weight_id = self.input_weight_layers[input_i][value_i].to_string();
                       let present_weight = self.weights_by_id.get(&weight_id).unwrap();

                       let new_weight = present_weight + rng.gen_range(LEARNING_RATE * -1., LEARNING_RATE);
                       self.weights_by_id.insert(weight_id, new_weight);

                       value_i += 1;
                   }

                   input_i += 1;
               }
        */

        // Mutate weights

        // for tuple in self.weights_by_id.iter() {
        //     let new_weight = tuple.1 + rng.gen_range(LEARNING_RATE * -1., LEARNING_RATE);
        //     self.weights_by_id.insert(*tuple.0, new_weight);
        // }

        // Not 100% sure this works
        for tuple in self.weights_by_id.iter_mut() {
            *tuple.1 += rng.gen_range(LEARNING_RATE * -1., LEARNING_RATE);
        }

        // Construct new weight layers

        let mut input_i = 0;
        while input_i < self.input_weight_layers.len() {
            let mut value_i = 0;
            while value_i < self.input_weight_layers[input_i].len() {
                let weight_id = self.input_weight_layers[input_i][value_i];
                let present_weight = self.weights_by_id.get(&weight_id).unwrap();

                self.weight_layers[0][input_i][value_i] = *present_weight;

                value_i += 1;
            }

            input_i += 1;
        }

        // Other layers

        let mut layer_i = 1;
        while layer_i < self.activation_layers.len() {
            let mut activation_index = 0;
            while activation_index < self.activation_layers[layer_i].len() {
                let mut weight_i = 0;

                while weight_i < self.weight_layers[layer_i][activation_index].len() {
                    self.weight_layers[layer_i][activation_index][weight_i] +=
                        rng.gen_range(LEARNING_RATE * -1., LEARNING_RATE);
                    weight_i += 1;
                }

                activation_index += 1;
            }

            layer_i += 1;
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

    pub unsafe fn clone(&self) -> NeuralNetwork {
        NeuralNetwork {
            id: NEURAL_NETWORK_MANAGER.lock().unwrap().new_id(),
            input_weight_layers: self.input_weight_layers.clone(),
            weights_by_id: self.weights_by_id.clone(),
            weight_layers: self.weight_layers.clone(),
            activation_layers: self.activation_layers.clone(),
        }
    }
}
