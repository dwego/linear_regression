use linear_regression::linear::LinearRegression;

fn main() {
    let lr = LinearRegression::fit(&[1.0, 2.0], &[2.0, 4.0]).unwrap();
}
