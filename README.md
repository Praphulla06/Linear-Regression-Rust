# Linear Regression in Rust

A simple **linear regression** implementation in Rust using `ndarray` for numerical operations and `ndarray-rand` for initializing weights. This project demonstrates training a linear model using gradient descent.

---

## Features

* Implements **simple linear regression** from scratch.
* Uses **mean squared error (MSE)** as the loss function.
* Trains with **gradient descent** to optimize weights and bias.
* Utilizes the `ndarray` crate for array operations.

---

## Installation

1. Make sure you have Rust installed. If not, install it via [rustup](https://www.rust-lang.org/tools/install).

2. Clone the repository:

```bash
git clone https://github.com/Praphulla06/Linear-Regression-Rust.git
cd linear_regression_rust
```

3. Build the project:

```bash
cargo build
```

4. Run the example:

```bash
cargo run
```

---

## Usage

The main example demonstrates a simple linear relationship:

```rust
use ndarray::Array;
use ndarray::prelude::*;

use linear_regression::LinearRegression;

fn main() {
    let x_train = Array::<f64, Ix1>::linspace(0., 2., 100);
    let y: Vec<f64> = x_train.to_vec().into_iter().map(|x| 2. * x + 5.).collect();
    let y_train = Array::from(y);

    println!("x_train: {}\ny_train: {}", x_train.len(), y_train.len());
    let mut model_0 = LinearRegression::new(&x_train, &y_train);
    
    println!(
        "weight: {:?}, bias: {:?}",
        model_0.weights.mean().unwrap(),
        model_0.bias
    );

    model_0.train(1);
    println!("Error before training: {:?}", model_0.error);

    let epoch = 1024;
    println!("Training...");
    model_0.train(epoch);

    println!("Error after training: {:?}. epochs: {}", model_0.error, epoch);
    println!(
        "weight: {:?}, bias: {:?}",
        model_0.weights.mean().unwrap(),
        model_0.bias
    );
}
```

**Output Example**:

```
x_train: 100
y_train: 100
weight: -0.012569690006533971, bias: 0.0
Error before training: 50.553554911307536
Training...
Error after training: 0.04073567862895372. epochs: 1024.
weight: 2.3294859523177642, bias: 4.609761319919488
```

---

## API

### `LinearRegression`

A struct representing a linear regression model:

```rust
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
```

### Methods

* `new(x_train: &Array<f64, Ix1>, y_train: &Array<f64, Ix1>) -> LinearRegression`
  Initializes a new linear regression model.

* `train(&mut self, epoch: i32)`
  Trains the model for a given number of epochs using gradient descent.

---

## Dependencies

* [`ndarray`](https://crates.io/crates/ndarray) — For multidimensional arrays.
* [`ndarray-rand`](https://crates.io/crates/ndarray-rand) — For random weight initialization.

```toml
[dependencies]
ndarray = "0.16.1"
ndarray-rand = "0.15.0"
rand = "0.9.2"
```

---

## Author

Created by **Praphulla P Pradhananga**.
