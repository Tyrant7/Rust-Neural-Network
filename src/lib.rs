#![allow(unused_imports)]
#![allow(dead_code)]

use std::{
    borrow::Borrow,
    collections::HashMap,
    time::{Duration, Instant},
    vec,
};

use std::sync::mpsc::Sender;

pub mod neural_network_old;
use layer::Layer;
use ndarray::{array, Array2};

use crate::neural_network_old::{NeuralNetwork_Old, NEURAL_NETWORK_MANAGER};

pub mod layer;
pub mod layers;
use layers::{activation_functions::{relu, relu_derivative, sigmoid, sigmoid_derivative}, linear::Linear};

pub mod neural_network;
use neural_network::NeuralNetwork;

const TICK_SPEED: u32 = 1;

pub fn main() {
    let mut network = NeuralNetwork::new(vec![
        Layer::linear(20, 20, relu, relu_derivative),
        Layer::linear(20, 10, sigmoid, sigmoid_derivative),
        Layer::linear(10, 3, relu, relu_derivative),
    ]); 
}


/*
pub fn run_ticks(neural_network: &mut NeuralNetwork_Old, inputs: Vec<f32>) {
    let time_start = Instant::now();

    for tick in 0..50000 {
        if tick > 500 {
            break;
        }

        print!("Processing tick: ");
        println!("{}", tick);

        let time_elapsed = time_start.elapsed();
        println!("{:?}", time_elapsed);

        let activations = neural_network.forward_propagate(inputs.clone());

        println!("Ouputs {:?}", activations.last().unwrap());

        if tick % 10 == 0 {
            neural_network.mutate();
            neural_network.write_to_file();
        }
    }
}
*/
