use ndarray::Array;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

struct LinearRegression {
    x_train: Array<f64, Ix1>,
    y_train: Array<f64, Ix1>,
    weights: Array<f64, Ix1>,
    bias: f64,
    lr: f64,
    y_pred: Array<f64, Ix1>,
    error: f64,
    residual: Array<f64, Ix1>,
}

impl LinearRegression {
    fn new(x_train: &Array<f64, Ix1>, y_train: &Array<f64, Ix1>) -> LinearRegression {
        LinearRegression {
            // x_train: (x_train - x_train.mean().unwrap()) / x_train.std(0.0),
            x_train: x_train.clone(),
            y_train: y_train.clone(),
            weights: Array::<f64, _>::random((1,), Normal::new(0., 0.01).unwrap()),
            bias: 0.0,
            lr: 0.01,
            y_pred: Array::<f64, Ix1>::zeros(x_train.len()),
            error: 0.0,
            residual: Array::<f64, Ix1>::zeros(x_train.len()),
        }
    }

    fn forward(&mut self) {
        self.y_pred = &self.weights * &self.x_train + self.bias;
    }

    fn residual(&mut self) {
        self.residual = &self.y_pred - &self.y_train;
    }

    fn mse(&mut self) {
        self.residual();
        let residual = self.residual.iter().map(|i| i.powf(2.0));
        let sum: f64 = residual.sum();
        self.error = sum / self.x_train.len() as f64;
    }

    fn gradient_descent(&mut self) {
        let grad_w = ((&self.residual * &self.x_train).sum()) / self.x_train.len() as f64;
        self.weights -= self.lr * grad_w;

        let grad_b = ((&self.residual).sum()) / self.x_train.len() as f64;
        self.bias -= self.lr * grad_b;
    }

    fn train(&mut self, epoch: i32) {
        for _ in 0..epoch {
            self.forward();
            self.residual();
            self.mse();
            self.gradient_descent();
        }
    }
}

fn main() {
    let x_train = Array::<f64, Ix1>::linspace(0., 2., 50);
    let y: Vec<f64> = x_train.to_vec().into_iter().map(|x| 2. * x + 5.).collect();
    let y_train = Array::from(y);

    println!("x_train: {}\ny_train: {}", x_train.len(), y_train.len());
    let mut model_0 = LinearRegression::new(&x_train, &y_train);

    model_0.mse();
    println!("Error: {:?}", model_0.error);
    println!(
        "weight: {:?}, bias: {:?}",
        model_0.weights.mean().unwrap(),
        model_0.bias
    );
    println!("Training...");
    model_0.train(1024);

    // println!("{:#?}", model_0.y_pred);
    println!("Error: {:?}", model_0.error);
    println!(
        "weight: {:?}, bias: {:?}",
        model_0.weights.mean().unwrap(),
        model_0.bias
    );
}
