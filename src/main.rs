#![allow(unused_imports)]
#![allow(dead_code)]

use std::{vec, collections::HashMap, borrow::Borrow, time::{Duration, Instant}};

use eventual::Timer;
use neural_network::{InputName, OutputName};
extern crate eventual;

use std::sync::mpsc::Sender;

mod neural_network;
use crate::neural_network::{Input, Output, NeuralNetwork,NEURAL_NETWORK_MANAGER};

const TICK_SPEED: u32 = 1;

fn main() {
    println!("Begin");

    /* let neural_network_manager = NeuralNetworkManager::new(); */

    let inputs: Vec<Input> = vec![
        Input::new(
            InputName::X,
            vec![1., 3.],
            vec![1, 2],
        ),
        Input::new(
            InputName::Y,
            vec![2.],
            vec![1],
        ),
    ];
    let outputs: Vec<Output> = vec![
        Output::new(OutputName::Result),
    ];
    
    let mut neural_network = init(&inputs, outputs.len());
    tick_manager(&mut neural_network, inputs);
    
    println!("End");
}

pub fn init(inputs: &Vec<Input>, output_count: usize) -> NeuralNetwork {
    
    let mut neural_network = NeuralNetwork::new();
    neural_network.build(inputs, output_count);

    neural_network
}

pub fn tick_manager(neural_network: &mut NeuralNetwork, inputs: Vec<Input>) {

    let time_start = Instant::now();

    let timer = Timer::new();
    let ticks = timer.interval_ms(TICK_SPEED).iter();
    
    for (tick, _) in ticks.enumerate() {

        if tick > 500 {
            break;
        }

        print!("Processing tick: ");
        println!("{}", tick);

        let time_elapsed = time_start.elapsed();
        println!("{:?}", time_elapsed);

        neural_network.forward_propagate(&inputs);

        if tick % 10 == 0 {

            neural_network.mutate();
            neural_network.write_to_file();
        }
    }
}