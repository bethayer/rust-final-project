use core::num;
use std::process::Output;

use nalgebra::DMatrix;
use rand::{Rng, rngs::adapter::ReseedingRng};

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
    output: bool,
    activation: Activation,
}

impl Layer {
    //initialize a new layer with random values
    fn new(input_size: usize, output_size: usize, output: bool, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();
        let weights = 
        DMatrix::from_fn(output_size, input_size, |_, _| rng.gen_range(-1.0_f64..=1.0_f64));
        let biases = DMatrix::zeros(output_size, 1);
        Layer {
            weights,
            biases,
            output,
            activation,
        }
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
        let mut z_s: Vec<DMatrix<f64>> = Vec::new();
        let mut activations: Vec<DMatrix<f64>> = Vec::new();

        let mut curr_input = input.clone();

        for layer in &self.layers {
            let (z, a) = layer.forward(&curr_input);
            z_s.push(z);
            activations.push(a.clone());
            curr_input = a;
        }

        (z_s, activations)
    }

    //takes in, the target values, the pre-activation values, and the post-activation values
    //outputs the weight gradients and the biases gradients
    fn back_prop(&self, target: &DMatrix<f64>, z: &DMatrix<f64>, a: &DMatrix<f64>) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
        todo!();
    }
    
    //takes in matrix of inputs where each colum is one sample (input_size * num_samples), matrix of targers where each colum is one target (output_size * num_samples), and the number of epochs
    pub fn train(&mut self, inputs: & DMatrix<f64>, targets: &DMatrix<f64>, epochs: usize) -> Result<(), String> {
        if self.layers.is_empty() {
            return Err(String::from("network has no layers"));
        }
        if inputs.ncols() != targets.ncols() {
            return Err(String::from("size of input not equal to size of targets"));
        }
        if inputs.ncols() == 0 {
            return Err(String::from("no training data"));
        }

        let num_samples = inputs.ncols();
        for _epoch in 0..epochs {
            for sample_idx in 0..num_samples {
                let input_sample = DMatrix::from_column_slice(
                    inputs.nrows(),
                    1,
                    inputs.column(sample_idx).as_slice(),
                );
                let target_sample = DMatrix::from_column_slice(
                    targets.nrows(),
                    1,
                    targets.column(sample_idx).as_slice(),
                );

                let (zs, activations) = self.forward_prop(&input_sample);

                let final_z = match zs.last() {
                    Some(z) => z.clone(),
                    None => return Err(String::from("forward prop returned no pre activations")),
                };
                let final_a = match activations.last() {
                    Some(a) => a.clone(),
                    None => return Err(String::from("forward prop returned no activations")),
                };

                let (weight_grads, bias_grads) = self.back_prop(&target_sample, &final_z, &final_a);

                // check if gradient count matches layer count
                if weight_grads.len() != self.layers.len() || bias_grads.len() != self.layers.len() {
                    return Err(String::from("gradient counts do not match"));
                }

                for (i, layer) in self.layers.iter_mut().enumerate() {
                    layer.weights -= self.alpha * &weight_grads[i];
                    layer.biases -= self.alpha * &bias_grads[i];
                }
            }
        }
        Ok(())
    }

    //passes though the network and give an output of the output nodes
    pub fn predict(&self, input: &DMatrix<f64>) -> Result<Vec<f64>, String> {
        let (_zs, activations) = self.forward_prop(input);
        match activations.last() {
            Some(output) => Ok(output.as_slice().to_vec()),
            None => Err(String::from("no layers in network")),
        }
    }
}