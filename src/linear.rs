pub struct LinearRegression {
    weight: f64,
    bias: f64
}

impl LinearRegression {
    pub fn fit (x: &[f64], y: &[f64]) -> Result<Self, &'static str>  {
        if x.len() != y.len() {
            return Err("x e y precisam ter o mesmo tamanho");
        }

        if x.len() < 2 {
            return Err("são necessários pelo menos dois pontos");
        }

        Ok(Self {
            weight: 2.0,
            bias: 0.0
        })
    }
}