#![allow(unused_imports)]
#![allow(dead_code)]

use std::{vec, collections::HashMap, borrow::Borrow, time::{Duration, Instant}};

use std::sync::mpsc::Sender;

pub mod neural_network;
pub mod utils;
use crate::neural_network::{NeuralNetwork,NEURAL_NETWORK_MANAGER};

const TICK_SPEED: u32 = 1;

fn main() {
    println!("Begin");

    /* let neural_network_manager = NeuralNetworkManager::new(); */

    let inputs: Vec<f64> = vec![1., 3.];
    let output_count = 5;
    
    let mut neural_network = init(inputs.len(), output_count);
    // tick_manager(&mut neural_network, inputs);
    
    println!("End");
}

pub fn init(input_count: usize, output_count: usize) -> NeuralNetwork {
    
    let neural_network = NeuralNetwork::new(0., 0.1, vec![input_count, 5, 3, output_count]);
    neural_network
}

// pub fn tick_manager(neural_network: &mut NeuralNetwork, inputs: Vec<Input>) {

//     let time_start = Instant::now();

//     let timer = Timer::new();
//     let ticks = timer.interval_ms(TICK_SPEED).iter();
    
//     for (tick, _) in ticks.enumerate() {

//         if tick > 500 {
//             break;
//         }

//         print!("Processing tick: ");
//         println!("{}", tick);

//         let time_elapsed = time_start.elapsed();
//         println!("{:?}", time_elapsed);

//         neural_network.forward_propagate(&inputs);

//         if tick % 10 == 0 {

//             neural_network.mutate();
//             neural_network.write_to_file();
//         }
//     }
// }