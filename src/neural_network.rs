pub struct NeuralNetworkManager {
    id_index: i32,
    networks: Vec<( i32, NeuralNetwork )>,
}

impl NeuralNetworkManager {
    pub fn init(&mut self) {

        return;
    }
    pub fn new_id(&mut self) {
        return self.id_index += 1;
    }
}


pub const neural_network_manager: NeuralNetworkManager = NeuralNetworkManager {
    id_index: 1,
    networks: vec![],
};


pub struct NeuralNetwork {
    id: i32,
    weight_layers: Vec<Vec<f32>>,
    activation_layers: Vec<Vec<f32>>,
}

struct Input {
    name: String,
    value: f32
}

struct Output {
    name: String,

}

impl NeuralNetwork {
    pub fn init(&mut self, weight_layers: Option<Vec<Vec<f32>>>, activation_layers: Option<Vec<Vec<f32>>>) {
        /* self.id = NeuralNetworkManager::new_id(NeuralNetworkManager); */

        /* if let Some(self.weight_layers) { self.weight_layers = weight_layers }; */
        if let Some(weight_layers) = weight_layers {
            self.weight_layers = weight_layers;
        };

        if let Some(activation_layers) = activation_layers {
            self.activation_layers = activation_layers
        }

        return
    }

    fn build(&mut self, input_count: i32, output_count: i32) {

        self.weight_layers.push(vec![]);
        self.activation_layers.push(vec![]);

        let mut i = -1;
        while i < input_count {

            i += 1;

            self.weight_layers[i as usize].push(i as f32);
        }
    }

    fn forward_propagate(&mut self, inputs: Vec<Input>) {

        for input in inputs {

            input.value;
        }
    }

    fn back_propagate(&mut self, scored_outputs: bool) {


    }

    fn mutate(&mut self) {


    }

    fn visualize(&mut self) {


    }
}