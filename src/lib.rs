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
use layers::linear::Linear;
use layers::relu::ReLU;
use layers::sigmoid::Sigmoid;

pub mod neural_network;
use neural_network::NeuralNetwork;

const TICK_SPEED: u32 = 1;

pub fn main() {
    println!("Begin");

    let inputs: Vec<f32> = vec![1., 3.];
    let output_count = 1;

    let mut neural_network = init(inputs.len(), output_count);
    run_ticks(&mut neural_network, inputs);

    println!("End");



    let mut network = create_network(vec![
        Linear::new(20, 20), 
        ReLU, 
        Linear::new(20, 10),
        Sigmoid,
        Linear::new(10, 3),
    ]);
}

pub fn create_network(layers: Vec<Layer>) -> NeuralNetwork_Old {
    let neural_network = NeuralNetwork_Old::new()
}

pub fn init(input_count: usize, output_count: usize) -> NeuralNetwork_Old {
    let neural_network = NeuralNetwork_Old::new(1., 0.1, vec![input_count, 5, 3, output_count]);
    neural_network
}

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
