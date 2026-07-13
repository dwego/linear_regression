use linear_regression::linear::LinearRegression;

fn main() {
    let lr = LinearRegression::fit_mse(&[1.0, 2.0], &[2.0, 4.0], 0.01, 100).unwrap();

    println!("Weight: {}", lr.weight);
    println!("Bias: {}", lr.bias);

}
