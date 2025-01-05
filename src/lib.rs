use std::vec;

pub mod layer;
pub mod optimizer;

use layer::Layer;

pub mod layers;
pub mod activation_functions;

use activation_functions::ActivationFunction::{ReLU, Sigmoid};

pub mod neural_network;
use ndarray::Array2;
use neural_network::NeuralNetwork;

pub mod optimizers;
use optimizer::Optimizer;
use optimizers::sgd::SGD;

pub fn main() {
    test_xor();
}

pub fn test_xor() {
    let mut network = NeuralNetwork::new(vec![
        Layer::linear(2, 2, ReLU),
        Layer::linear(2, 1, ReLU),
    ]);

    let mut optimizer = SGD { learning_rate: 0.001 };

    // All inputs of XOR matched to their respective outputs
    let train_data = [
        ([0., 0.], [0.]), 
        ([0., 1.], [1.]),
        ([1., 0.], [1.]),
        ([1., 1.], [0.])
    ];

    println!("Beginning training a network to solve XOR problem....");

    let GENERATIONS = 50;
    for i in (0..GENERATIONS) {
        let mut generation_error = 0;

        for (inputs, expected) in train_data.iter() {
            let activations = network.forward(Vec::from(inputs));
            let gradients = network.backwards(&activations, Vec::from(expected));

            // Calculate mean absolute error for analysis
            let final_output = activations.last().unwrap();
            let targets_array = Array2::from_shape_fn((expected.len(), 1), |(j, _k)| expected[j]);
            generation_error += (final_output - targets_array).abs();

            optimizer.update(&mut network, &gradients);
        }

        generation_error /= train_data.len();
        println!("Generation {} error: {}", i, generation_error);
    }
}

pub fn sample() {
    let mut network = NeuralNetwork::new(vec![
        Layer::linear(4, 4, ReLU),
        Layer::linear(4, 2, Sigmoid),
        Layer::linear(2, 1, ReLU),
    ]);

    let inputs: Vec<f32> = vec![0., 0., 0., 0.];
    let target: f32 = 1.;

    println!("\nData initialized:");
    println!("inputs:");
    println!("{:?}", inputs);

    let activations = network.forward(inputs);

    println!("\nActivations:");
    println!("{:?}", activations);
    
    println!("output:");
    println!("{:?}", activations.last().unwrap());
    println!("target:");
    println!("{}", target);

    println!("\nBeginning backward pass...");

    let gradients = network.backwards(&activations, vec![target]);

    println!("Gradients:");
    println!("{:#?}", gradients);

    println!("Optimizing gradients...");
    let mut optimizer = SGD { learning_rate: 0.001 };
    optimizer.update(&mut network, &gradients);

    println!("\nAnalysis complete!");
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
