#![allow(unused_imports)]
#![allow(dead_code)]

mod neural_network;
use neural_network::{NeuralNetwork, NeuralNetworkManager};

use crate::neural_network::{Input, Output};

static mut NEURAL_NETWORK_MANAGER: NeuralNetworkManager = NeuralNetworkManager {
    id_index: 1,
    networks: vec![],
    bias: 1,
    learning_rate: 1,
    hidden_layers_count: 2,
    hidden_perceptron_count: 3,
};

fn main() {
    println!("Hello, world!");

    /* let neural_network_manager = NeuralNetworkManager::new(); */


}

fn init() {

    let inputs: Vec<Input> = vec![
        Input {
            name: "x".to_string(),
            value: 1,
            weightID: "1".to_string(),
        },
        Input {
            name: "y".to_string(),
            value: 1,
            weightID: "1".to_string(),
        },
    ];
    let outputs: Vec<Output> = vec![
        Output {
            name: "result".to_string(),
        },
    ];
/* 
    let neural_network = NeuralNetwork {
        id: NeuralNetworkManager::new_id(),
        ..Default::default()
    }
    neural_network.init()
     */
}