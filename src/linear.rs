pub struct LinearRegression {
    weight: f64,
    bias: f64
}

impl LinearRegression {
    pub fn fit_mse (x: &[f64], y: &[f64]) -> Result<Self, &'static str>  {
        if x.len() != y.len() {
            return Err("x e y precisam ter o mesmo tamanho");
        }

        if x.len() < 2 {
            return Err("são necessários pelo menos dois pontos");
        }

        let initial_weight = 0.0;
        let initial_bias = 0.0;

        let loss_func_derivate_weight = 2.0
            * x.iter()
                .zip(y)
                .map(|(&x_i, &y_i)| (initial_weight * x_i + initial_bias - y_i) * x_i)
                .sum::<f64>()
            / x.len() as f64;

        let loss_func_derivate_bias = 2.0
            * x.iter()
                .zip(y)
                .map(|(&x_i, &y_i)| initial_weight * x_i + initial_bias - y_i)
                .sum::<f64>()
            / x.len() as f64;

        Ok(Self {
            weight: 2.0,
            bias: 0.0
        })
    }
}