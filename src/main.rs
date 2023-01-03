#![allow(unused_imports)]
#![allow(dead_code)]

mod neural_network;
use std::{vec, collections::HashMap, borrow::Borrow};

use crate::neural_network::{Input, Output, NeuralNetwork,NEURAL_NETWORK_MANAGER};
/* 
static mut NEURAL_NETWORK_MANAGER: NeuralNetworkManager = NeuralNetworkManager {
    id_index: 1,
    networks: vec![],
};
 */
fn main() {
    println!("Hello, world!");

    /* let neural_network_manager = NeuralNetworkManager::new(); */

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

    let mut neural_network = init(&inputs, outputs.len());
    neural_network.forward_propagate(&inputs.to_vec());

/* 
    for tuple in NEURAL_NETWORK_MANAGER.lock().unwrap().networks {

        let neural_network = tuple.1;
        neural_network.forward_propagate(&inputs);
    }
     */
}

fn init(inputs: &Vec<Input>, output_count: usize) -> NeuralNetwork {

    let mut neural_network = NeuralNetwork {
        id: NEURAL_NETWORK_MANAGER.lock().unwrap().new_id(),
        input_weights: HashMap::new(),
        input_weight_layers: vec![],
        weight_layers: vec![],
        activation_layers: vec![],
    };
    neural_network.build(inputs, output_count);

    return neural_network;
}