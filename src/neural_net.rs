use nalgebra::DMatrix;
use rand::Rng;

pub enum Activation{
    ReLu,
}

impl Activation {
    pub fn apply_activation(&self, z: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            Activation::ReLu => {
                z.map(|v| v.max(0.0))
            }
        }
    }

    pub fn derivative(&self, z: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            Activation::ReLu => {
                z.map(|v| if v > 0.0 {1.0} else {0.0})
            }
        }
    }
}

struct Layer {
    weights: DMatrix<f64>,
    biases: DMatrix<f64>,
    //if it is an output layer, true, then we don't need to apply the activation function
    output: bool
}

impl Layer {
    //initialize a new layer with random values
    fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        todo!()
    }

    //first output is the preactivation-values, second output is the post-activation values
    //z = W*x + b, a = activation(z)
    fn forward(&self, input: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
        todo!()
    } 
}

pub struct NeuralNetwork {
    layers: Vec<Layer>,
    //learning rate
    alpha: f64
}

impl NeuralNetwork {
    //initialize a new neural network of a number of hidden layes, where each hidden layer has the same number of nodes
    pub fn new(input_size: usize, num_hidden_layers: usize, hidden_layer_size: usize, output_size: usize, alpha: f64) -> Self {
        todo!()
    }

    //first output is the list of containing the layers before activation and second is the list of layers after activation
    fn forward_prop(&self, input: & DMatrix<f64>) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
        todo!();
    }

    //takes in, the target values, the pre-activation values, and the post-activation values
    //outputs the weight gradients and the biases gradients
    fn back_prop(&self, target: &DMatrix<f64>, z: &DMatrix<f64>, a: &DMatrix<f64>) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
        todo!();
    }
    
    //takes in matrix of inputs where each colum is one sample (input_size * num_samples), matrix of targers where each colum is one target (output_size * num_samples), and the number of epochs
    pub fn train(&mut self, inputs: & DMatrix<f64>, targets: &DMatrix<f64>, epochs: usize) {
        todo!();
    }

    //passes though the network and give an output of the output nodes
    pub fn predict(&self, input: &DMatrix<f64>) -> Vec<f64> {
        todo!();
    }
}