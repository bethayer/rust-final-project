use nalgebra::DMatrix;
use rand::Rng;

pub enum Activation {
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
                z.map(|v| if v > 0.0 { 1.0 } else { 0.0 })
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
    //initialize a new layer with He initialization
    //weights are scaled by sqrt(2 / input_size) to keep pre-activation values
    //in a reasonable range, which is critical for ReLU networks to train well
    fn new(input_size: usize, output_size: usize, output: bool, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / input_size as f64).sqrt();
        let weights = DMatrix::from_fn(output_size, input_size, |_, _| {
            rng.gen_range(-1.0_f64..=1.0_f64) * scale
        });
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
        let z = &self.weights * input + &self.biases;
        let mut a = z.clone();
        if !self.output {
            a = self.activation.apply_activation(&z);
        }
        return (z, a);
    }
}

pub struct NeuralNetwork {
    layers: Vec<Layer>,
    //learning rate
    alpha: f64,
}

impl NeuralNetwork {
    //initialize a new neural network of a number of hidden layes, where each hidden layer has the same number of nodes
    pub fn new(input_size: usize, num_hidden_layers: usize, hidden_layer_size: usize, output_size: usize, alpha: f64) -> Self {
        todo!()
    }

    //first output is the list of containing the layers before activation and second is the list of layers after activation
    fn forward_prop(&self, input: &DMatrix<f64>) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
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

    //takes in the input, the target values, all pre-activation values, and all post-activation values
    //outputs the weight gradients and the biases gradients
    fn back_prop(&self, input: &DMatrix<f64>, target: &DMatrix<f64>, zs: &Vec<DMatrix<f64>>, activations: &Vec<DMatrix<f64>>) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
        todo!()
    }

    //takes in matrix of inputs where each colum is one sample (input_size * num_samples), matrix of targers where each colum is one target (output_size * num_samples), and the number of epochs
    pub fn train(&mut self, inputs: &DMatrix<f64>, targets: &DMatrix<f64>, epochs: usize) -> Result<(), String> {
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
        for epoch in 0..epochs {
            for sample_idx in 0..num_samples {
                if sample_idx % 5000 == 0 {
                    println!("  epoch {}/{} - sample {}/{}", epoch + 1, epochs, sample_idx, num_samples);
                }

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

                let (weight_grads, bias_grads) = self.back_prop(&input_sample, &target_sample, &zs, &activations);

                // check if gradient count matches layer count
                if weight_grads.len() != self.layers.len() || bias_grads.len() != self.layers.len() {
                    return Err(String::from("gradient counts do not match"));
                }

                for (i, layer) in self.layers.iter_mut().enumerate() {
                    layer.weights -= self.alpha * &weight_grads[i];
                    layer.biases -= self.alpha * &bias_grads[i];
                }
            }
            println!("Epoch {}/{} complete", epoch + 1, epochs);
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