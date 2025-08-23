use ndarray::Array;
use ndarray::prelude::*;

use linear_regression::LinearRegression;

fn main() {
    let x_train = Array::<f64, Ix1>::linspace(0., 2., 50);
    let y: Vec<f64> = x_train.to_vec().into_iter().map(|x| 2. * x + 5.).collect();
    let y_train = Array::from(y);

    println!("x_train: {}\ny_train: {}", x_train.len(), y_train.len());
    let mut model_0 = LinearRegression::new(&x_train, &y_train);

    println!("Error: {:?}", model_0.error);
    println!(
        "weight: {:?}, bias: {:?}",
        model_0.weights.mean().unwrap(),
        model_0.bias
    );

    println!("Training...");
    model_0.train(1024);

    println!("Error: {:?}", model_0.error);
    println!(
        "weight: {:?}, bias: {:?}",
        model_0.weights.mean().unwrap(),
        model_0.bias
    );
}
