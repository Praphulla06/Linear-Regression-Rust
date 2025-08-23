use ndarray::Array;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

pub struct LinearRegression {
    pub x_train: Array<f64, Ix1>,
    pub y_train: Array<f64, Ix1>,
    pub weights: Array<f64, Ix1>,
    pub bias: f64,
    pub lr: f64,
    pub y_pred: Array<f64, Ix1>,
    pub error: f64,
    pub residual: Array<f64, Ix1>,
}

impl LinearRegression {
    pub fn new(x_train: &Array<f64, Ix1>, y_train: &Array<f64, Ix1>) -> LinearRegression {
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
        // self.residual();
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

    pub fn train(&mut self, epoch: i32) {
        for _ in 0..epoch {
            self.forward();
            self.residual();
            self.mse();
            self.gradient_descent();
        }
    }
}