use std::vec;

pub mod layer;
pub mod optimizer;

use layer::Layer;

pub mod activation_functions;
pub mod layers;

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
    let mut network =
        NeuralNetwork::new(vec![Layer::linear(2, 2, ReLU),  Layer::linear(2, 1, ReLU)]);

    let mut optimizer = SGD {
        learning_rate: 0.01,
    };

    // All inputs of XOR matched to their respective outputs
    let train_data = [
        ([0., 0.], [0.]),
        ([0., 1.], [1.]),
        ([1., 0.], [1.]),
        ([1., 1.], [0.]),
    ];

    println!("Beginning training a network to solve XOR problem....");

    for generation in 0..10000 {
        let mut generation_error = 0.;

        // Accumulate gradients for a whole generation before applying any changes to network parameters
        let mut gen_grads: Vec<(Array2<f32>, Array2<f32>)> = vec![];

        for layer in network.layers.iter() {
            match layer {
                Layer::Linear(linear) => {

                    gen_grads.push((
                        Array2::from_elem(linear.weights.raw_dim(), 0.),
                        Array2::from_elem(linear.bias.raw_dim(), 0.),
                    ))
                }
            }
        }

        for (inputs, expected) in train_data.iter() {

            let inputs_array = Array2::from_shape_vec((inputs.len(), 1), inputs.to_vec()).unwrap();

            let activations = network.forward(&inputs_array);
            let gradients = network.backwards(&activations, &inputs_array, Vec::from(expected));

            // Calculate mean absolute error for analysis
            let final_output = activations.last().unwrap();
            println!("output {} target {}", final_output.first().unwrap(), expected[0]);
            let targets_array = Array2::from_shape_fn((expected.len(), 1), |(j, _k)| expected[j]);
            generation_error += (final_output - targets_array).abs().sum();
            println!("error {generation_error}");

            for (layer_i, (weights, biases)) in gradients.iter().enumerate() {

                let layer_batch_grads = &mut gen_grads[layer_i];

                layer_batch_grads.0 += weights;
                layer_batch_grads.1 += biases;
            }
        }

        // Optimize parameters
        optimizer.update(&mut network, &gen_grads);

        generation_error /= train_data.len() as f32;
        println!("Generation {} error: {}", generation, generation_error);
    }
}

pub fn sample() {
    let mut network = NeuralNetwork::new(vec![
        Layer::linear(4, 4, ReLU),
        Layer::linear(4, 2, Sigmoid),
        Layer::linear(2, 1, ReLU),
    ]);

    let inputs_vec: Vec<f32> = vec![0., 0., 0., 0.];
    let inputs = Array2::from_shape_vec((4, 1), inputs_vec).unwrap();
    let target: f32 = 1.;

    println!("\nData initialized:");
    println!("inputs:");
    println!("{:?}", inputs);

    let activations = network.forward(&inputs);

    println!("\nActivations:");
    println!("{:?}", activations);

    println!("output:");
    println!("{:?}", activations.last().unwrap());
    println!("target:");
    println!("{}", target);

    println!("\nBeginning backward pass...");

    let gradients = network.backwards(&activations, &inputs, vec![target]);

    println!("Gradients:");
    println!("{:#?}", gradients);

    println!("Optimizing gradients...");
    let mut optimizer = SGD {
        learning_rate: 0.001,
    };
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
