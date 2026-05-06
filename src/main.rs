mod neural_net;
use neural_net::NeuralNetwork;
use nalgebra::DMatrix;
use mnist_reader::MnistReader;
use rand::Rng;
use std::error::Error;
use std::io::{self, Write};

//convert mnist_reader Vec<Vec<f32>> images and Vec<u8> labels into the column-major DMatrix<f64> shapes
fn to_matrices(images: &[Vec<f32>], labels: &[u8]) -> (DMatrix<f64>, DMatrix<f64>) {
    let num_samples = images.len();

    let inputs = DMatrix::from_fn(784, num_samples, |row, col| {
        f64::from(images[col][row]) / 255.0
    });

    let targets = DMatrix::from_fn(10, num_samples, |row, col| {
        if row == labels[col] as usize { 1.0 } else { 0.0 }
    });

    (inputs, targets)
}

//run inference on every column of input, compare against one-hot targets, and return (correct, total).
fn evaluate(nn: &NeuralNetwork, inputs: &DMatrix<f64>, targets: &DMatrix<f64>) -> (usize, usize) {
    let mut correct = 0;
    let total = inputs.ncols();

    for i in 0..total {
        let sample = inputs.column(i);
        let sample_matrix = DMatrix::from_column_slice(784, 1, sample.as_slice());
        let output = nn.predict(&sample_matrix).unwrap();

        let predicted = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        let actual = (0..10).find(|&j| targets[(j, i)] == 1.0).unwrap();
        if predicted == actual {
            correct += 1;
        }
    }

    (correct, total)
}

//print a prompt and read a line of trimmed input from stdin
fn prompt(message: &str) -> io::Result<String> {
    print!("{}", message);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

//prompt until the user enters a valid value of type T
fn prompt_parse<T: std::str::FromStr>(message: &str) -> T {
    loop {
        match prompt(message) {
            Ok(s) => match s.parse::<T>() {
                Ok(v) => return v,
                Err(_) => println!("  invalid input, try again"),
            },
            Err(_) => println!("  could not read input, try again"),
        }
    }
}

//build a fresh NeuralNetwork from user defined parameters
fn build_network_from_user_input() -> NeuralNetwork {
    println!("\n--- Network setup ---");
    let num_hidden_layers: usize = prompt_parse("Number of hidden layers: ");
    let hidden_layer_size: usize = prompt_parse("Neurons per hidden layer: ");
    let alpha: f64 = prompt_parse("Learning rate: ");
    NeuralNetwork::new(784, num_hidden_layers, hidden_layer_size, 10, alpha)
}

//train an existing network for the given number of epochs and report accuracy
fn run_training(
    nn: &mut NeuralNetwork,
    train_inputs: &DMatrix<f64>,
    train_targets: &DMatrix<f64>,
    test_inputs: &DMatrix<f64>,
    test_targets: &DMatrix<f64>,
    epochs: usize,
    total_epochs: &mut usize,
) -> Result<(), Box<dyn Error>> {
    println!("\nTraining for {} epoch(s)...", epochs);
    nn.train(train_inputs, train_targets, epochs)?;
    *total_epochs += epochs;
    println!("Training complete. Total epochs this session: {}", total_epochs);

    let (tc, tt) = evaluate(nn, train_inputs, train_targets);
    println!(
        "Training accuracy: {}/{} ({:.1}%)",
        tc, tt, tc as f64 / tt as f64 * 100.0
    );

    let (sc, st) = evaluate(nn, test_inputs, test_targets);
    println!(
        "Test accuracy:     {}/{} ({:.1}%)",
        sc, st, sc as f64 / st as f64 * 100.0
    );

    Ok(())
}

//pick a random test sample, run the network on it, and print prediction vs actual
fn run_prediction(
    nn: &NeuralNetwork,
    test_inputs: &DMatrix<f64>,
    test_targets: &DMatrix<f64>,
) {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(0..test_inputs.ncols());

    let sample = test_inputs.column(i);
    let sample_matrix = DMatrix::from_column_slice(784, 1, sample.as_slice());
    let output = nn.predict(&sample_matrix).unwrap();

    let predicted = output
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    let actual = (0..10).find(|&j| test_targets[(j, i)] == 1.0).unwrap();

    println!("\n--- Prediction ---");
    println!("Test sample index: {}", i);
    println!("Predicted: {}", predicted);
    println!("Actual:    {}", actual);
    println!("{}", if predicted == actual { "Correct!" } else { "Wrong." });
}

fn main() -> Result<(), Box<dyn Error>> {
    // Load MNIST once at startup.
    let mut mnist = MnistReader::new("mnist-data");
    mnist.load()?;
    println!(
        "Loaded MNIST: {} training samples, {} test samples",
        mnist.train_data.len(),
        mnist.test_data.len()
    );

    let (train_inputs, train_targets) = to_matrices(&mnist.train_data, &mnist.train_labels);
    let (test_inputs,  test_targets)  = to_matrices(&mnist.test_data,  &mnist.test_labels);

    // Session state.
    let mut nn: Option<NeuralNetwork> = None;
    let mut total_epochs: usize = 0;

    // Initial choice: train or predict.
    loop {
        let action = prompt("\nWhat would you like to do? [train / predict / quit]: ")?;
        match action.as_str() {
            "train" => {
                // First training run sets up the network.
                if nn.is_none() {
                    nn = Some(build_network_from_user_input());
                }
                let epochs: usize = prompt_parse("Number of epochs to train: ");
                run_training(
                    nn.as_mut().unwrap(),
                    &train_inputs, &train_targets,
                    &test_inputs,  &test_targets,
                    epochs, &mut total_epochs,
                )?;
            }
            "predict" => {
                match nn.as_ref() {
                    Some(net) => run_prediction(net, &test_inputs, &test_targets),
                    None => println!("No network yet — train first."),
                }
            }
            "quit" | "exit" => {
                println!("Bye.");
                break;
            }
            _ => println!("Unknown command. Try 'train', 'predict', or 'quit'."),
        }
    }

    Ok(())
}