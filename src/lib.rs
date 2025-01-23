use core::error;
use std::vec;

pub mod layer;
pub mod optimizer;

use layer::Layer;

pub mod activation_functions;
pub mod layers;

use activation_functions::ActivationFunction::{ReLU, Sigmoid};

pub mod neural_network;
use colored::Colorize;
use ndarray::Array2;
use neural_network::NeuralNetwork;

pub mod optimizers;
use optimizer::Optimizer;
use optimizers::sgd::SGD;
use plotters::{prelude::{BitMapBackend, IntoDrawingArea}, series::AreaSeries, style::{Color, BLUE, WHITE}};

pub fn main() {
    test_xor();
}

pub fn test_xor() {
    let mut network =
        NeuralNetwork::new(vec![Layer::linear(2, 2, ReLU), Layer::linear(2, 1, ReLU)]);

    println!("NEURAL NETWORK {:?}", network.layers);

    let mut optimizer = SGD { learning_rate: 0.1 };

    // All inputs of XOR matched to their respective outputs
    // let train_data = [
    //     ([0., 0.], [0.]),
    //     ([0., 1.], [1.]),
    //     ([1., 0.], [1.]),
    //     ([1., 1.], [0.]),
    // ];
    let inputs_serial: Vec<Array2<f32>> = vec![
        Array2::from_shape_vec((1, 2), vec![0., 0.]).unwrap(),
        Array2::from_shape_vec((1, 2), vec![0., 1.]).unwrap(),
        Array2::from_shape_vec((1, 2), vec![1., 0.]).unwrap(),
        Array2::from_shape_vec((1, 2), vec![1., 1.]).unwrap(),
    ];
    let targets_serial: Vec<Array2<f32>> = vec![
        Array2::from_shape_vec((1, 1), vec![0.]).unwrap(),
        Array2::from_shape_vec((1, 1), vec![1.]).unwrap(),
        Array2::from_shape_vec((1, 1), vec![1.]).unwrap(),
        Array2::from_shape_vec((1, 1), vec![0.]).unwrap(),
    ];

    let inputs_homogenous =
        Array2::from_shape_vec((4, 2), vec![0., 0., 0., 1., 1., 0., 1., 1.]).unwrap();
    let targets_homogenous = Array2::from_shape_vec((4, 1), vec![0., 1., 1., 0.]).unwrap();

    println!("Beginning training a network to solve XOR problem....");

    let mut errors = vec![];

    for generation in 0..1000 {
        let mut generation_error = 0.;

        // Accumulate gradients for a whole generation before applying any changes to network parameters
        let mut gen_weight_gradients: Vec<Array2<f32>> = vec![];
        let mut gen_bias_gradients: Vec<Array2<f32>> = vec![];

        for layer in network.layers.iter() {
            match layer {
                Layer::Linear(linear) => {
                    gen_weight_gradients.push(Array2::from_elem(linear.weights.raw_dim(), 0.));
                    gen_bias_gradients.push(Array2::from_elem(linear.bias.raw_dim(), 0.));
                }
            }
        }

        // let inputs = inputs_homogenous.clone();
        // let targets = targets_homogenous.clone();

        for (inputs, targets) in inputs_serial.iter().zip(&targets_serial) {
            let (activations, transfers) = network.forward(inputs);
            let (weight_gradients, bias_gradients) =
                network.backwards(&activations, &transfers, inputs, targets);

            // Calculate mean absolute error for analysis
            let final_output = activations.last().unwrap();

            generation_error += (final_output - targets).abs().sum();
            // let mse_error = (targets - final_output).mapv(|x| x * x).sum();
            // generation_error += mse_error;
            // println!("{} {}", "MSE".red().bold(), mse_error);

            // optimizer.update(&mut network, &gradients);

            for (layer_i, (weights, biases)) in
                (weight_gradients).iter().zip(&bias_gradients).enumerate()
            {
                let gen_layer_weight_gradients = &mut gen_weight_gradients[layer_i];
                let gen_layer_bias_gradients = &mut gen_bias_gradients[layer_i];

                *gen_layer_weight_gradients += weights;
                *gen_layer_bias_gradients += biases;
            }
        }

        // for (inputs, expected) in train_data.iter() {

        //     let inputs_array = Array2::from_shape_vec((inputs.len(), 1), inputs.to_vec()).unwrap();

        //     let (activations, transfers) = network.forward(&inputs_array);
        //     let gradients = network.backwards(&activations, &transfers, &inputs_array, Vec::from(expected));

        //     // Calculate mean absolute error for analysis
        //     let final_output = activations.last().unwrap();
        //     println!("output {} target {}", final_output.first().unwrap(), expected[0]);
        //     let targets_array = Array2::from_shape_fn((expected.len(), 1), |(j, _k)| expected[j]);
        //     generation_error += (final_output - &targets_array).abs().sum();

        //     // optimizer.update(&mut network, &gradients);

        //     for (layer_i, (weights, biases)) in gradients.iter().enumerate() {

        //         let layer_batch_grads = &mut gen_grads[layer_i];

        //         layer_batch_grads.0 += weights;
        //         layer_batch_grads.1 += biases;
        //     }
        // }

        // Optimize parameters
        optimizer.update(&mut network, &gen_weight_gradients, &gen_bias_gradients);

        generation_error /= inputs_serial.len() as f32;
        println!(
            "{} {} {} {}",
            "Generation:".bright_green(),
            generation,
            "Error:".bright_red(),
            generation_error
        );

        errors.push(generation_error);
    }

    let _ = simple_chart(errors, "Error", "charts");
}

fn simple_chart(data: Vec<f32>, name: &str, dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/{name}.png");
    let root = BitMapBackend::new(path.as_str(), (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_min = data
    .iter()
    .min_by(|a, b| a.partial_cmp(b).unwrap())
    .unwrap()
    * 0.9;

let y_max = data
    .iter()
    .max_by(|a, b| a.partial_cmp(b).unwrap())
    .unwrap()
    * 1.1;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption(name, ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..data.len() as u32, y_min..y_max)?;

    chart.configure_mesh().light_line_style(WHITE).draw()?;

    chart.draw_series(
        AreaSeries::new(
            data.iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value)),
            0.0,
            BLUE.mix(0.2),
        )
        .border_style(BLUE),
    )?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
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
    let target_array = Array2::from_shape_vec((1, 1), vec![target]).unwrap();

    println!("\nData initialized:");
    println!("inputs:");
    println!("{:?}", inputs);

    let (activations, transfers) = network.forward(&inputs);

    println!("\nActivations:");
    println!("{:?}", activations);

    println!("output:");
    println!("{:?}", activations.last().unwrap());
    println!("target:");
    println!("{}", target);

    println!("\nBeginning backward pass...");

    let (weight_gradients, bias_gradients) =
        network.backwards(&activations, &transfers, &inputs, &target_array);

    println!("Optimizing gradients...");
    let mut optimizer = SGD {
        learning_rate: 0.001,
    };
    optimizer.update(&mut network, &weight_gradients, &bias_gradients);

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
