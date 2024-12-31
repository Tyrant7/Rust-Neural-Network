use ndarray::Array2;

pub fn relu(input: Array2<f32>) -> Array2<f32> {
    input.mapv(|x| x.max(0.))        
}

pub fn relu_derivative(activations: Array2<f32>) -> Array2<f32> {
    activations.mapv(|x| if x > 0. { 1. } else { 0. })
}

pub fn sigmoid(input: Array2<f32>) -> Array2<f32> {
    input.mapv(sigmoid_internal)
}

pub fn sigmoid_derivative(activations: Array2<f32>) -> Array2<f32> {
    activations.mapv(|x| sigmoid_internal(x) * (1. - sigmoid_internal(x)))
}

fn sigmoid_internal(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}