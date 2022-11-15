use std::sync::Mutex;

pub struct NeuralNetworkManager {
    id_index: i32,
    networks: Vec<( i32, NeuralNetwork )>,
    hidden_layers_count: usize,
    hidden_perceptron_count: usize,
}

impl NeuralNetworkManager {
    pub fn init(&mut self) {

        return;
    }
    pub fn new_id(&mut self) -> i32 {
        
        self.id_index += 1;
        return self.id_index
    }
}

static neural_network_manager: Mutex<NeuralNetworkManager> = Mutex::new(NeuralNetworkManager {
    id_index: 1,
    networks: vec![],
    hidden_layers_count: 2,
    hidden_perceptron_count: 3,
});

fn a() {
    neural_network_manager.lock();
}

/* 
static neural_network_manager: Mutex<NeuralNetworkManager> = NeuralNetworkManager {
    id_index: 1,
    networks: vec![],
    hidden_layers_count: 2,
    hidden_perceptron_count: 3,
};
 */
/* 
neural_network_manager: NeuralNetworkManager = NeuralNetworkManager {
    id_index: 1,
    networks: vec![],
    hidden_layers_count: 2,
    hidden_perceptron_count: 3,
};
 */

pub struct NeuralNetwork {
    id: i32,
    weight_layers: Vec<Vec<Vec<f32>>>,
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
    pub fn init(&mut self, weight_layers: Option<Vec<Vec<Vec<f32>>>>, activation_layers: Option<Vec<Vec<f32>>>) {
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

    fn build(&mut self, input_count: usize, output_count: usize) {

        self.weight_layers.push(vec![]);
        self.activation_layers.push(vec![]);

        // Construct the input layer

        let mut i = 0;
        while i < input_count {

            self.weight_layers[i as usize].push(vec![i as f32]);
            self.activation_layers[i as usize].push(i as f32);

            i += 1;
        }

        // Construct hidden layers

        let mut i1 = 0;
        while i1 < neural_network_manager.hidden_layers_count {

            self.weight_layers.push(vec![]);
            self.activation_layers.push(vec![]);

            let mut i2 = 0;
            while i2 < neural_network_manager.hidden_perceptron_count {

                self.weight_layers[i1 as usize].push(vec![]);

                let mut i3 = 0;
                while i3 < self.activation_layers[(i1 - 1) as usize].len() {

                    self.weight_layers[i1 as usize][i2 as usize].push(0.);

                    i3 += 1;
                }

                self.activation_layers[i1 as usize].push(0.);

                i2 += 1;
            }

            i1 += 1;
        }

        self.weight_layers.push(vec![]);
        self.activation_layers.push(vec![]);

        let last_layer_index = self.activation_layers.len() - 1;

        i1 = 0;
        while i < output_count {

            self.weight_layers[last_layer_index].push(vec![]);

            let mut i2 = 0;
            while i2 < self.activation_layers[last_layer_index - 1].len() {

                self.weight_layers[last_layer_index][i1].push(0.);

                i2 += 1;
            }

            self.activation_layers[last_layer_index].push(0.);

            i1 += 1;
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

    fn clone(&self) -> NeuralNetwork {

        return NeuralNetwork {
            id: neural_network_manager.new_id(),
            weight_layers: self.weight_layers.to_vec(),
            activation_layers: self.activation_layers.to_vec(),
        };
    }
}