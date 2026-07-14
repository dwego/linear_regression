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
        Self::fit_with_gradient(x, y, alpha, epochs, batch_size, |error| {
            error.signum()
        })
    }

    pub fn fit_mse(
        x: &Matrix,
        y: &[f64],
        alpha: f64,
        epochs: usize,
        batch_size: usize,
    ) -> Result<Self, &'static str> {
        Self::fit_with_gradient(x, y, alpha, epochs, batch_size, |error| {
            2.0 * error
        })
    }

    fn fit_with_gradient(
        x: &Matrix,
        y: &[f64],
        alpha: f64,
        epochs: usize,
        batch_size: usize,
        gradient_factor: impl Fn(f64) -> f64,
    ) -> Result<Self, &'static str> {
        Self::validate_training_input(x, y, alpha, epochs, batch_size)?;

        let mut weight = vec![0.0; x.cols()];
        let mut bias = 0.0;

        for _ in 0..epochs {
            let mut batch_start = 0;

            while batch_start < x.rows() {
                let batch_end = (batch_start + batch_size).min(x.rows());

                Self::update_batch(
                    x,
                    y,
                    alpha,
                    batch_start,
                    batch_end,
                    &mut weight,
                    &mut bias,
                    &gradient_factor,
                );

                batch_start = batch_end;
            }
        }

        Ok(Self { weight, bias })
    }

    fn update_batch(
        x: &Matrix,
        y: &[f64],
        alpha: f64,
        batch_start: usize,
        batch_end: usize,
        weight: &mut [f64],
        bias: &mut f64,
        gradient_factor: &impl Fn(f64) -> f64,
    ) {
        let mut dw = vec![0.0; x.cols()];
        let mut db = 0.0;

        for row in batch_start..batch_end {
            let prediction = Self::prediction_for_row(x, row, weight, *bias);
            let error = prediction - y[row];
            let factor = gradient_factor(error);

            for col in 0..x.cols() {
                dw[col] += factor * x[[row, col].into()];
            }

            db += factor;
        }

        let batch_len = (batch_end - batch_start) as f64;

        for col in 0..x.cols() {
            weight[col] -= alpha * (dw[col] / batch_len);
        }

        *bias -= alpha * (db / batch_len);
    }

    fn prediction_for_row(
        x: &Matrix,
        row: usize,
        weight: &[f64],
        bias: f64,
    ) -> f64 {
        weight
            .iter()
            .enumerate()
            .map(|(col, &w)| w * x[[row, col].into()])
            .sum::<f64>()
            + bias
    }

    fn validate_training_input(
        x: &Matrix,
        y: &[f64],
        alpha: f64,
        epochs: usize,
        batch_size: usize,
    ) -> Result<(), &'static str> {
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

        Ok(())
    }

    pub fn predict(&self, x: &Matrix) -> Result<Vec<f64>, &'static str> {
        if x.cols() != self.weight.len() {
            return Err("x must have the same number of columns as the model");
        }

        let predictions = (0..x.rows())
            .map(|row| Self::prediction_for_row(x, row, &self.weight, self.bias))
            .collect();

        Ok(predictions)
    }
}