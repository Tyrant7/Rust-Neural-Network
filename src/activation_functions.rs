use ndarray::Array2;

#[derive(Debug)]
pub enum ActivationFunction {
    None,
    ReLU,
    Sigmoid,
    LeakyReLU
}

impl ActivationFunction {
    pub fn plain(&self, input: Array2<f32>) -> Array2<f32> {
        match self {
            ActivationFunction::None => input,
            ActivationFunction::ReLU => input.mapv(|x| x.max(0.)),
            ActivationFunction::Sigmoid => input.mapv(sigmoid_internal),
            ActivationFunction::LeakyReLU => input.mapv(|x| match x < 0. {
                true => 0.01 * x,
                false => x
            }),
        }
    }

    pub fn derivative(&self, input: Array2<f32>) -> Array2<f32> {
        match self {
            ActivationFunction::None => input,
            ActivationFunction::ReLU => input.mapv(|x| if x > 0. { 1. } else { 0. }),
            ActivationFunction::Sigmoid => input.mapv(|x| x * (1. - x)/* sigmoid_internal(x) * (1. - sigmoid_internal(x)) */),
            ActivationFunction::LeakyReLU => input.mapv(|x| match x < 0. {
                true => 0.01,
                false => 1.
            })
        }
    }
}

fn sigmoid_internal(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
