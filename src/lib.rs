pub mod layer;
use layer::Layer;

pub mod layers;
use layers::activation_functions::ActivationFunction::{ReLU, Sigmoid};

pub mod neural_network;
use neural_network::NeuralNetwork;

const TICK_SPEED: u32 = 1;

pub fn main() {
    let mut network = NeuralNetwork::new(vec![
        Layer::linear(20, 20, ReLU),
        Layer::linear(20, 10, Sigmoid),
        Layer::linear(10, 3, ReLU),
    ]); 

    println!("It runs!");
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
