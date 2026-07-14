pub mod linear;

#[cfg(test)]
mod tests {
    use crate::linear::LinearRegression;
    use num_rs::Matrix;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }

    #[test]
    fn predict_should_return_one_prediction_per_matrix_row() {
        let model = LinearRegression {
            weight: vec![3.0, 2.0],
            bias: 5.0,
        };

        let x = Matrix::new(
            3,
            2,
            vec![
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
            ],
        );

        let predictions = model.predict(&x.unwrap()).unwrap();

        assert_eq!(predictions.len(), 3);

        assert_close(predictions[0], 12.0, 1e-6);
        assert_close(predictions[1], 22.0, 1e-6);
        assert_close(predictions[2], 32.0, 1e-6);
    }

    #[test]
    fn predict_should_return_error_when_column_count_does_not_match_model() {
        let model = LinearRegression {
            weight: vec![3.0, 2.0],
            bias: 5.0,
        };

        let x = Matrix::new(
            1,
            3,
            vec![
                1.0, 2.0, 3.0,
            ],
        );

        let result = model.predict(&x.unwrap());

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "x must have the same number of columns as the model"
        );
    }

    #[test]
    fn fit_mse_should_learn_single_feature_linear_relation() {
        let x = Matrix::new(
            5,
            1,
            vec![
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
        );

        let y = vec![
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
        ];

        let model = LinearRegression::fit_mse(&x.unwrap(), &y, 0.01, 10_000, 5).unwrap();

        assert_eq!(model.weight.len(), 1);
        assert_close(model.weight[0], 2.0, 0.01);
        assert_close(model.bias, 0.0, 0.05);

        let x_predict = Matrix::new(
            3,
            1,
            vec![
                6.0,
                7.0,
                8.0,
            ],
        );

        let predictions = model.predict(&x_predict.unwrap()).unwrap();

        assert_close(predictions[0], 12.0, 0.1);
        assert_close(predictions[1], 14.0, 0.1);
        assert_close(predictions[2], 16.0, 0.1);
    }

    #[test]
    fn fit_mse_should_learn_multiple_feature_linear_relation() {
        let x = Matrix::new(
            6,
            2,
            vec![
                1.0, 1.0,
                2.0, 1.0,
                1.0, 2.0,
                2.0, 2.0,
                3.0, 2.0,
                2.0, 3.0,
            ],
        );

        let y = vec![
            10.0,
            13.0,
            12.0,
            15.0,
            18.0,
            17.0,
        ];

        let model = LinearRegression::fit_mse(&x.unwrap(), &y, 0.01, 10_000, 6).unwrap();

        assert_eq!(model.weight.len(), 2);

        assert_close(model.weight[0], 3.0, 0.1);
        assert_close(model.weight[1], 2.0, 0.1);
        assert_close(model.bias, 5.0, 0.2);

        let x_predict = Matrix::new(
            2,
            2,
            vec![
                4.0, 3.0,
                3.0, 4.0,
            ],
        );

        let predictions = model.predict(&x_predict.unwrap()).unwrap();

        assert_eq!(predictions.len(), 2);

        assert_close(predictions[0], 23.0, 0.3);
        assert_close(predictions[1], 22.0, 0.3);
    }

    #[test]
    fn fit_mse_should_learn_house_price_example() {
        let x_train = Matrix::new(
            8,
            2,
            vec![
                1.0, 5.0,
                2.0, 6.0,
                2.0, 8.0,
                3.0, 8.0,
                3.0, 10.0,
                4.0, 10.0,
                4.0, 12.0,
                5.0, 14.0,
            ],
        );

        let y_train = vec![
            63_000.0,
            68_000.0,
            72_000.0,
            75_000.0,
            79_000.0,
            82_000.0,
            86_000.0,
            93_000.0,
        ];

        let model =
            LinearRegression::fit_mse(&x_train.unwrap(), &y_train, 0.001, 100_000, 8).unwrap();

        assert_eq!(model.weight.len(), 2);


        assert_close(model.weight[0], 3_000.0, 1.0);
        assert_close(model.weight[1], 2_000.0, 1.0);
        assert_close(model.bias, 50_000.0, 2.0);

        let x_predict = Matrix::new(
            3,
            2,
            vec![
                3.0, 12.0,
                4.0, 15.0,
                2.0, 7.0,
            ],
        );

        let predictions = model.predict(&x_predict.unwrap()).unwrap();

        assert_eq!(predictions.len(), 3);

        assert_close(predictions[0], 83_000.0, 2.0);
        assert_close(predictions[1], 92_000.0, 2.0);
        assert_close(predictions[2], 70_000.0, 2.0);
    }

    #[test]
    fn fit_mae_should_train_and_return_valid_model() {
        let x = Matrix::new(
            5,
            1,
            vec![
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
        );

        let y = vec![
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
        ];

        let model = LinearRegression::fit_mae(&x.unwrap(), &y, 0.01, 10_000, 5).unwrap();

        assert_eq!(model.weight.len(), 1);


        let x_predict = Matrix::new(
            2,
            1,
            vec![
                6.0,
                7.0,
            ],
        );

        let predictions = model.predict(&x_predict.unwrap()).unwrap();

        assert_eq!(predictions.len(), 2);

        assert_close(predictions[0], 12.0, 1.0);
        assert_close(predictions[1], 14.0, 1.0);
    }

    #[test]
    fn fit_mse_should_return_error_when_x_and_y_have_different_lengths() {
        let x = Matrix::new(
            3,
            1,
            vec![
                1.0,
                2.0,
                3.0,
            ],
        );

        let y = vec![
            2.0,
            4.0,
        ];

        let result = LinearRegression::fit_mse(&x.unwrap(), &y, 0.01, 100, 3);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "x and y must have the same number of rows"
        );
    }

    #[test]
    fn fit_mse_should_return_error_when_less_than_two_rows() {
        let x = Matrix::new(
            1,
            1,
            vec![
                1.0,
            ],
        );

        let y = vec![
            2.0,
        ];

        let result = LinearRegression::fit_mse(&x.unwrap(), &y, 0.01, 100, 1);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "at least two data points are required"
        );
    }

    #[test]
    fn fit_mse_should_return_error_when_alpha_is_zero() {
        let x = Matrix::new(
            2,
            1,
            vec![
                1.0,
                2.0,
            ],
        );

        let y = vec![
            2.0,
            4.0,
        ];

        let result = LinearRegression::fit_mse(&x.unwrap(), &y, 0.0, 100, 2);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "alpha must be greater than zero"
        );
    }

    #[test]
    fn fit_mse_should_return_error_when_alpha_is_negative() {
        let x = Matrix::new(
            2,
            1,
            vec![
                1.0,
                2.0,
            ],
        );

        let y = vec![
            2.0,
            4.0,
        ];

        let result = LinearRegression::fit_mse(&x.unwrap(), &y, -0.01, 100, 2);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "alpha must be greater than zero"
        );
    }

    #[test]
    fn fit_mse_should_return_error_when_epochs_is_zero() {
        let x = Matrix::new(
            2,
            1,
            vec![
                1.0,
                2.0,
            ],
        );

        let y = vec![
            2.0,
            4.0,
        ];

        let result = LinearRegression::fit_mse(&x.unwrap(), &y, 0.01, 0, 2);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "epochs must be greater than zero"
        );
    }

    #[test]
    fn fit_mae_should_return_same_validation_errors_as_fit_mse() {
        let x = Matrix::new(
            2,
            1,
            vec![
                1.0,
                2.0,
            ],
        );

        let y = vec![
            2.0,
        ];

        let result = LinearRegression::fit_mae(&x.unwrap(), &y, 0.01, 100, 2);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "x and y must have the same number of rows"
        );
    }

    #[test]
    fn fit_mse_should_return_error_when_batch_size_is_zero() {
        let x = Matrix::new(
            2,
            1,
            vec![
                1.0,
                2.0,
            ],
        );

        let y = vec![
            2.0,
            4.0,
        ];

        let result = LinearRegression::fit_mse(&x.unwrap(), &y, 0.01, 100, 0);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "batch_size must be greater than zero"
        );
    }
}