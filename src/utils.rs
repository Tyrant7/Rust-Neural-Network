pub fn relu(transfer: f32) -> f32 {
    transfer.max(0.)
}

pub fn relu_derivative(transfer: f32) -> f32 {
    match transfer > 0. {
        true => 1.,
        false => 0.,
    }
}

pub fn sigmoid(transfer: f32) -> f32 {
    1. / (1. + (-transfer).exp())
}

pub fn sigmoid_derivative(transfer: f32) -> f32 {
    sigmoid(transfer) * (1. - sigmoid(transfer))
}