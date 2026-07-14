use num_rs::Matrix;

#[derive(Debug)]
pub struct LinearRegression {
    pub weight: Vec<f64>,
    pub bias: f64,
}

impl LinearRegression {
    pub fn fit_mae(
        x: &Matrix,
        y: &[f64],
        alpha: f64,
        epochs: usize,
        batch_size: usize,
    ) -> Result<Self, &'static str> {
        let (mut weight, mut bias, _) =
            Self::fit(x, y, alpha, epochs, batch_size)?;

        for _epoch in 0..epochs {
            let mut batch_start = 0;

            while batch_start < x.rows() {
                let batch_end = (batch_start + batch_size).min(x.rows());

                let mut dw = vec![0.0; x.cols()];
                let mut db = 0.0;

                for row in batch_start..batch_end {
                    let yi = y[row];

                    let prediction = weight
                        .iter()
                        .enumerate()
                        .map(|(col, &w)| w * x[[row, col].into()])
                        .sum::<f64>()
                        + bias;

                    let error = prediction - yi;

                    let error_sign = if error > 0.0 {
                        1.0
                    } else if error < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };

                    for col in 0..x.cols() {
                        dw[col] += error_sign * x[[row, col].into()];
                    }

                    db += error_sign;
                }

                let current_batch_size = (batch_end - batch_start) as f64;

                for col in 0..x.cols() {
                    dw[col] /= current_batch_size;
                    weight[col] -= alpha * dw[col];
                }

                db /= current_batch_size;
                bias -= alpha * db;

                batch_start = batch_end;
            }
        }

        Ok(Self { weight, bias })
    }

    pub fn fit_mse(
        x: &Matrix,
        y: &[f64],
        alpha: f64,
        epochs: usize,
        batch_size: usize,
    ) -> Result<Self, &'static str> {
        let (mut weight, mut bias, _) = match Self::fit(&x, y, alpha, epochs, batch_size) {
            Ok(value) => value,
            Err(value) => return Err(value),
        };

        for _epoch in 0..epochs {
            let mut batch_start = 0;

            while batch_start < x.rows() {
                let batch_end = (batch_start + batch_size).min(x.rows());

                let mut dw = vec![0.0; x.cols()];
                let mut db = 0.0;

                for row in batch_start..batch_end {
                    let prediction = weight
                        .iter()
                        .enumerate()
                        .map(|(col, &w)| w * x[[row, col].into()])
                        .sum::<f64>()
                        + bias;

                    let error = prediction - y[row];

                    for col in 0..x.cols() {
                        dw[col] += 2.0 * error * x[[row, col].into()];
                    }

                    db += 2.0 * error;
                }

                let batch_len = (batch_end - batch_start) as f64;

                for col in 0..x.cols() {
                    dw[col] /= batch_len;
                    weight[col] -= alpha * dw[col];
                }

                db /= batch_len;
                bias -= alpha * db;

                batch_start = batch_end;
            }
        }

        Ok(Self { weight, bias })
    }

    fn fit(
        x: &Matrix,
        y: &[f64],
        alpha: f64,
        epochs: usize,
        batch_size: usize,
    ) -> Result<(Vec<f64>, f64, f64), &'static str> {
        if x.rows() != y.len() {
            return Err("x and y must have the same number of rows");
        }

        if x.rows() < 2 {
            return Err("at least two data points are required");
        }

        if x.cols() == 0 {
            return Err("x must have at least one column");
        }

        if alpha <= 0.0 {
            return Err("alpha must be greater than zero");
        }

        if epochs == 0 {
            return Err("epochs must be greater than zero");
        }

        if batch_size == 0 {
            return Err("batch_size must be greater than zero");
        }

        let weight = vec![0.0; x.cols()];
        let bias = 0.0;
        let n = x.rows() as f64;

        Ok((weight, bias, n))
    }

    pub fn predict(&self, x: &Matrix) -> Result<Vec<f64>, &'static str> {
        if x.cols() != self.weight.len() {
            return Err("x must have the same number of columns as the model");
        }

        let mut predictions = Vec::with_capacity(x.rows());

        for row in 0..x.rows() {
            let prediction = self
                .weight
                .iter()
                .enumerate()
                .map(|(col, &w)| w * x[[row, col].into()])
                .sum::<f64>()
                + self.bias;

            predictions.push(prediction);
        }

        Ok(predictions)
    }
}