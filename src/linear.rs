pub struct LinearRegression {
    pub weight: f64,
    pub bias: f64,
}

impl LinearRegression {
    pub fn fit_mse(
        x: &[f64],
        y: &[f64],
        alpha: f64,
        epochs: usize,
    ) -> Result<Self, &'static str> {
        if x.len() != y.len() {
            return Err("x e y precisam ter o mesmo tamanho");
        }

        if x.len() < 2 {
            return Err("são necessários pelo menos dois pontos");
        }

        if alpha <= 0.0 {
            return Err("alpha precisa ser maior que zero");
        }

        if epochs == 0 {
            return Err("epochs precisa ser maior que zero");
        }

        let mut weight = 0.0;
        let mut bias = 0.0;

        let n = x.len() as f64;

        for _ in 0..epochs {
            let loss_derivative_weight = 2.0
                * x.iter()
                .zip(y.iter())
                .map(|(&x_i, &y_i)| {
                    let prediction = weight * x_i + bias;
                    let error = prediction - y_i;

                    error * x_i
                })
                .sum::<f64>()
                / n;

            let loss_derivative_bias = 2.0
                * x.iter()
                .zip(y.iter())
                .map(|(&x_i, &y_i)| {
                    let prediction = weight * x_i + bias;

                    prediction - y_i
                })
                .sum::<f64>()
                / n;

            weight -= alpha * loss_derivative_weight;
            bias -= alpha * loss_derivative_bias;
        }

        Ok(Self { weight, bias })
    }

    pub fn predict(&self, x: f64) -> f64 {
        self.weight * x + self.bias
    }
}