pub fn relu(transfer: f64) -> f64 {
    transfer.max(0.)
}

pub fn relu_derivative(transfer: f64) -> f64 {
    match transfer > 0. {
        true => 1.,
        false => 0.,
    }
}

pub fn sigmoid(transfer: f64) -> f64 {
    1. / (1. + (-transfer).exp())
}

pub fn sigmoid_derivative(transfer: f64) -> f64 {
    sigmoid(transfer) * (1. - sigmoid(transfer))
}