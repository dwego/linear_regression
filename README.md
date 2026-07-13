# Linear Regression from Scratch (Rust)

A from-scratch, dependency-free implementation of linear regression in Rust — no ML frameworks, every piece of math implemented by hand. This README explains the theory behind each piece and walks through exactly how the code implements it. It's a living document: it grows alongside the repository, so at any point in time it only covers what has actually been implemented so far.

## Summary

1. [What is Linear Regression?](#1-what-is-linear-regression)
2. [Measuring the Error: Mean Squared Error (MSE)](#2-measuring-the-error-mean-squared-error-mse)
3. [Finding the Minimum: Gradients](#3-finding-the-minimum-gradients)
4. [Gradient Descent: The Training Algorithm](#4-gradient-descent-the-training-algorithm)
5. [How This Maps to the Code](#5-how-this-maps-to-the-code)
6. [Running the Project](#6-running-the-project)

## 1. What is Linear Regression?

Linear regression models the relationship between an input variable `x` and an output variable `y` as a straight line:

$$
\hat{y} = wx + b
$$

- `w` (**weight** / slope) — how much `y_pred` changes per unit of `x`.
- `b` (**bias** / intercept) — the value of `y_pred` when `x = 0`.

Given a dataset of `n` observed pairs `(x_i, y_i)`, the goal is to find the values of `w` and `b` that make the line fit the data as closely as possible. "As closely as possible" needs a precise definition — that's what the **loss function** provides.

This project implements the *simple* (single-feature) case: one input, one output, two parameters (`w`, `b`).

## 2. Measuring the Error: Mean Squared Error (MSE)

For a given `w` and `b`, each data point has a **prediction error**:

$$
e_i = (wx_i + b) - y_i
$$

The **Mean Squared Error** loss aggregates all errors into a single number that tells us how bad the current line is:

$$
MSE(w, b) = \frac{1}{n} \sum_{i=1}^{n} (wx_i + b - y_i)^2
$$

Squaring the error does two things:
1. Makes all errors positive (so they don't cancel out).
2. Penalizes large errors more than small ones, which pushes the fit toward the average trend rather than being dominated by noise.

`MSE(w, b)` is a smooth, bowl-shaped (convex) function of `w` and `b`. That convexity is what makes gradient descent reliable here — there's a single global minimum, no risk of getting stuck in a bad local minimum.

## 3. Finding the Minimum: Gradients

To minimize `MSE(w, b)`, we need to know which direction decreases it fastest. That's the job of the **gradient** — the vector of partial derivatives with respect to each parameter.

### Partial derivative with respect to `w`

$$
\frac{\partial MSE}{\partial w} = \frac{2}{n} \sum_{i=1}^{n} (wx_i + b - y_i) \, x_i = \frac{2}{n} \sum_{i=1}^{n} e_i x_i
$$

### Partial derivative with respect to `b`

$$
\frac{\partial MSE}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (wx_i + b - y_i) = \frac{2}{n} \sum_{i=1}^{n} e_i
$$

**Intuition:**
- $\partial MSE/\partial w$ scales the error by $x_i$ — points farther from the origin on the x-axis have more leverage on the slope.
- $\partial MSE/\partial b$ is just the average error — the intercept shifts up or down until predictions are centered on the data.

Both derivatives point in the direction of **steepest increase** of the loss. To reduce the loss, we move the parameters in the *opposite* direction.

## 4. Gradient Descent: The Training Algorithm

Gradient descent is an iterative optimization algorithm. Starting from an initial guess, it repeatedly nudges the parameters a small step against the gradient, gradually walking down the loss "bowl" toward the minimum.

**Algorithm:**

1. Initialize `w = 0`, `b = 0`.
2. Repeat for a fixed number of `epochs`:
   a. Compute predictions for every data point using the current `w`, `b`.
   b. Compute $\partial MSE/\partial w$ and $\partial MSE/\partial b$ over the *entire* dataset (this is **batch** gradient descent — every data point contributes to every update).
   c. Update the parameters against the gradient, scaled by a **learning rate** $\alpha$:

$$
w \leftarrow w - \alpha \frac{\partial MSE}{\partial w}
\qquad
b \leftarrow b - \alpha \frac{\partial MSE}{\partial b}
$$

3. After `epochs` iterations, return the final `(w, b)` as the trained model.

**Why subtract?** The gradient points uphill (direction of increasing loss), so subtracting it moves `w` and `b` downhill, toward lower loss.

**Role of `alpha` (the learning rate):**
- Too small → convergence is correct but very slow (many epochs needed).
- Too large → updates overshoot the minimum and the loss can oscillate or diverge instead of decreasing.
- `alpha` and `epochs` together control the trade-off between training speed and stability.

Each full pass over the dataset that produces one parameter update is called an **epoch**. Since this is batch gradient descent, one epoch = one gradient computation = one update (as opposed to stochastic/mini-batch variants, which update multiple times per epoch on subsets of the data).

## 5. How This Maps to the Code

### `LinearRegression` struct — [src/linear.rs](src/linear.rs)

```rust
pub struct LinearRegression {
    pub weight: f64,
    pub bias: f64,
}
```

Holds the trained parameters `w` (`weight`) and `b` (`bias`) from the equations above.

### `fit_mse` — training via gradient descent

```rust
pub fn fit_mse(x: &[f64], y: &[f64], alpha: f64, epochs: usize) -> Result<Self, &'static str>
```

This is the training loop described in Section 4:

1. **Validation** — ensures `x` and `y` have equal, sufficient length, and that `alpha`/`epochs` are valid (a learning rate of `0` would never update anything; `0` epochs would return an untrained model).
2. **Initialization** — `weight` and `bias` start at `0.0`.
3. **Loop over `epochs`** — on every iteration:
   - `loss_derivative_weight` computes $\partial MSE/\partial w = \frac{2}{n}\sum (\text{prediction} - y_i) \, x_i$ by mapping over every `(x_i, y_i)` pair, computing `prediction = weight * x_i + bias`, then `error = prediction - y_i`, multiplying by `x_i`, and averaging.
   - `loss_derivative_bias` computes $\partial MSE/\partial b = \frac{2}{n}\sum (\text{prediction} - y_i)$ the same way, without the `x_i` factor.
   - Both gradients are computed using the *current* `weight`/`bias` **before** either is updated — otherwise the bias gradient would be computed with an already-updated weight, corrupting the math.
   - The update step `weight -= alpha * loss_derivative_weight` and `bias -= alpha * loss_derivative_bias` is exactly the gradient descent update rule from Section 4.
4. After all epochs, the fitted `weight` and `bias` are returned wrapped in `Self`.

### `predict` — using the trained model

```rust
pub fn predict(&self, x: f64) -> f64 {
    self.weight * x + self.bias
}
```

A direct implementation of `y_pred = w * x + b` from Section 1, using the parameters learned during training.

### `main.rs` — example usage

```rust
let lr = LinearRegression::fit_mse(&[1.0, 2.0], &[2.0, 4.0], 0.01, 100).unwrap();
println!("Weight: {}", lr.weight);
println!("Bias: {}", lr.bias);
```

Trains on two points that lie perfectly on the line `y = 2x` (i.e. the true relationship is `w = 2`, `b = 0`). With `alpha = 0.01` and `100` epochs, gradient descent should converge close to those values — running it with more epochs or a larger `alpha` (while still small enough to remain stable) will push the result even closer to the exact solution.

## 6. Running the Project

```bash
cargo run          # runs the example in main.rs
cargo test          # if/when tests are added
```
