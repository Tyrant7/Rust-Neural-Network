pub struct NeuralNetworkManager {
    id_index: i32,
    networks:NeuralNetwork,
}

impl NeuralNetworkManager {
    pub fn init(&mut self) {

        return;
    }
    pub fn new_id(&mut self) {
        return self.id_index += 1;
    }
}


/* let neural_network_manager = NeuralNetworkManager::new(); */

pub struct NeuralNetwork {
    id: i32
}

impl NeuralNetwork {
    pub fn init(&mut self) {
        /* self.id = NeuralNetworkManager::new_id(NeuralNetworkManager); */
    }
}

