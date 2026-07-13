# Linear Regression from Scratch (Rust)

A from-scratch implementation of linear regression in Rust — no ML frameworks, every piece of math implemented by hand. This README explains the theory behind each piece and walks through exactly how the code implements it. It's a living document: it grows alongside the repository, so at any point in time it only covers what has actually been implemented so far.

## Summary

1. [What is Linear Regression?](#1-what-is-linear-regression)
2. [Measuring the Error: Loss Functions](#2-measuring-the-error-loss-functions)
3. [Finding the Minimum: Gradients](#3-finding-the-minimum-gradients)
4. [Gradient Descent: The Training Algorithm](#4-gradient-descent-the-training-algorithm)
5. [How This Maps to the Code](#5-how-this-maps-to-the-code)
6. [Unit Tests](#6-unit-tests)
7. [Running the Project](#7-running-the-project)

## 1. What is Linear Regression?

Linear regression models an output `y` as a weighted sum of one or more input features, plus a constant offset. For a single data point `i` with features $x_{i,1}, x_{i,2}, \dots, x_{i,m}$:

$$
\hat{y}_i = w_1 x_{i,1} + w_2 x_{i,2} + \dots + w_m x_{i,m} + b = \sum_{j=1}^{m} w_j x_{i,j} + b
$$

Or, more compactly, as a dot product between the feature row $x_i$ and the weight vector $w$:

$$
\hat{y}_i = w \cdot x_i + b
$$

- $w$ (**weight vector**) — one coefficient per feature, controlling how much that feature contributes to the prediction.
- $b$ (**bias**) — the value of $\hat{y}$ when every feature is `0`.
- $m$ — number of features (columns).
- $n$ — number of data points (rows).

The whole dataset is a matrix `X` with `n` rows (samples) and `m` columns (features), so the input **is not required to be single-dimensional** — `m` can be `1` (a single feature, the simplest case) or many. The math and the code don't special-case `m = 1`; a one-column matrix is just a regular case of the general formula above.

## 2. Measuring the Error: Loss Functions

For any `w` and `b`, each data point has a **prediction error**:

$$
e_i = \hat{y}_i - y_i = (w \cdot x_i + b) - y_i
$$

A **loss function** aggregates all `n` errors into a single number that says how bad the current `(w, b)` is. This project currently implements two.

### 2.1 Mean Squared Error (MSE)

$$
MSE(w, b) = \frac{1}{n} \sum_{i=1}^{n} e_i^2
$$

Squaring the error does two things:
1. Makes all errors positive (so they don't cancel out).
2. Penalizes large errors much more than small ones — a single big outlier can dominate the loss.

`MSE` is smooth and convex in `(w, b)`, which makes its gradient well-defined everywhere and gradient descent very predictable.

### 2.2 Mean Absolute Error (MAE)

$$
MAE(w, b) = \frac{1}{n} \sum_{i=1}^{n} |e_i|
$$

Instead of squaring, MAE takes the absolute value of each error. This means every unit of error contributes proportionally to the loss, regardless of its size — a large outlier hurts the loss no more, relatively, than several small errors. As a result, MAE is **more robust to outliers** than MSE. The trade-off is that `|e|` is not differentiable at `e = 0`, so training near a perfect fit is less smooth (see the gradient below).

## 3. Finding the Minimum: Gradients

To minimize a loss with gradient descent we need its partial derivative with respect to every parameter: each weight component $w_j$ and the bias $b$.

### 3.1 Gradients of MSE

$$
\frac{\partial MSE}{\partial w_j} = \frac{2}{n} \sum_{i=1}^{n} e_i \, x_{i,j}
\qquad
\frac{\partial MSE}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} e_i
$$

### 3.2 Gradients of MAE

The derivative of $|e|$ with respect to $e$ is the **sign function**:

$$
\frac{d|e|}{de} = \operatorname{sign}(e) =
\begin{cases}
+1 & e > 0 \\
-1 & e < 0 \\
0 & e = 0
\end{cases}
$$

(`0` at `e = 0` isn't a "true" derivative — `|e|` has a sharp corner there — but it's the conventional **subgradient** choice, and it's what this codebase uses: a perfectly predicted point contributes nothing to the update.)

Applying the chain rule through $e_i = w \cdot x_i + b - y_i$:

$$
\frac{\partial MAE}{\partial w_j} = \frac{1}{n} \sum_{i=1}^{n} \operatorname{sign}(e_i) \, x_{i,j}
\qquad
\frac{\partial MAE}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} \operatorname{sign}(e_i)
$$

Unlike MSE's gradient, which shrinks as the error shrinks (proportional to `e`), MAE's gradient has constant magnitude regardless of how close the prediction is — only its *sign* depends on the error. That's why training with MAE tends to be noisier close to the minimum.

## 4. Gradient Descent: The Training Algorithm

Gradient descent is an iterative optimization algorithm. Starting from an initial guess, it repeatedly nudges every parameter a small step against its gradient, walking down the loss surface toward the minimum.

**Algorithm (generalized to `m` features):**

1. Initialize $w_1, \dots, w_m = 0$ and $b = 0$.
2. Repeat for a fixed number of `epochs`:
   a. For every data point $i$, compute the prediction $\hat{y}_i = w \cdot x_i + b$ and the error $e_i = \hat{y}_i - y_i$.
   b. Accumulate the gradient of the chosen loss with respect to every $w_j$ and $b$, summed over **all** `n` rows (this is **batch** gradient descent).
   c. Divide the accumulated sums by `n`, then update every parameter against its gradient, scaled by a **learning rate** $\alpha$:

$$
w_j \leftarrow w_j - \alpha \frac{\partial \, \text{loss}}{\partial w_j} \quad \text{for every } j
\qquad
b \leftarrow b - \alpha \frac{\partial \, \text{loss}}{\partial b}
$$

3. After `epochs` iterations, return the final `(w, b)` as the trained model.

**Role of $\alpha$ (the learning rate):**
- Too small → convergence is correct but very slow.
- Too large → updates overshoot the minimum and the loss can oscillate or diverge.

## 5. How This Maps to the Code

### Matrix input — [src/linear.rs](src/linear.rs)

`x` is represented as a [`num_rs::Matrix`](Cargo.toml) (an external dependency), with `x.rows()` samples and `x.cols()` features, indexed as `x[[row, col].into()]`. This is what makes the model **not** limited to a single feature — a `Matrix` with one column behaves exactly like `m` columns everywhere in the code, no special casing needed.

### `LinearRegression` struct

```rust
pub struct LinearRegression {
    pub weight: Vec<f64>,
    pub bias: f64,
}
```

`weight` is now a `Vec<f64>` — one entry per feature/column — instead of a single `f64`, matching the generalized $w$ from Section 1.

### `fit` — shared validation and initialization

```rust
fn fit(x: &Matrix, y: &[f64], alpha: f64, epochs: usize) -> Result<(Vec<f64>, f64, f64), &'static str>
```

Both `fit_mse` and `fit_mae` delegate their setup to this private helper, so the two losses share identical validation behavior:

- `x.rows() == y.len()` — one target per sample.
- `x.rows() >= 2` — at least two data points.
- `x.cols() > 0` — at least one feature/column.
- `alpha > 0` and `epochs > 0`.

It returns a zero-initialized `weight` vector sized `x.cols()`, `bias = 0.0`, and `n = x.rows() as f64`, matching Step 1 of the algorithm in Section 4.

### `fit_mse` — training with MSE

Implements the MSE gradient descent loop from Section 3.1 / 4: for every epoch, it loops over every row, computes the prediction as a dot product (`weight.iter().enumerate().map(|(col, &w)| w * x[[row, col]]).sum()` `+ bias`), the error, and accumulates `dw[col] += 2.0 * error * x[[row, col]]` and `db += 2.0 * error`. After the row loop, both are divided by `n` and used to update `weight[col]` and `bias` against the gradient, scaled by `alpha`.

### `fit_mae` — training with MAE

Implements the MAE gradient descent loop from Section 3.2 / 4. It's structurally the same as `fit_mse`, but instead of accumulating `2.0 * error`, it computes `error_sign` (`+1.0`, `-1.0`, or `0.0` for `error == 0.0`) and accumulates `dw[col] += error_sign * x[[row, col]]` and `db += error_sign` — the sign-function gradient from Section 3.2. It also tracks the mean absolute error each epoch and prints `epoch`, `loss`, `weight`, and `bias` every 100 epochs, useful for watching convergence (or the lack of it) during training.

### `predict` — using the trained model

```rust
pub fn predict(&self, x: &Matrix) -> Result<Vec<f64>, &'static str>
```

Validates that `x.cols() == self.weight.len()` (a model trained on `m` features can't predict on a matrix with a different number of columns), then returns one prediction per row, each computed as the dot product $w \cdot x_i + b$ from Section 1.

## 6. Unit Tests

Tests live in [src/lib.rs](src/lib.rs), covering:

- **`predict`** — returns exactly one prediction per matrix row on a hand-computed multi-feature model; returns an error when the input matrix's column count doesn't match the model's weight length.
- **`fit_mse`** — learns a single-feature relation (`y = 2x`); learns a two-feature linear relation; learns a more realistic multi-feature "house price" example with larger-magnitude targets, verifying both the learned `weight`/`bias` and the resulting predictions land within tolerance.
- **`fit_mae`** — trains on the same single-feature data as `fit_mse` and produces a valid model with reasonable predictions.
- **Shared validation** — `fit_mse` is checked to return the correct error for mismatched `x`/`y` row counts, fewer than two rows, `alpha == 0`, `alpha < 0`, and `epochs == 0`; `fit_mae` is checked to return the same validation error as `fit_mse` for mismatched row counts, confirming both losses share the same `fit` validation path.

## 7. Running the Project

This is currently a library crate (no `main.rs` binary) — the way to exercise it is through its test suite:

```bash
cargo test
```
