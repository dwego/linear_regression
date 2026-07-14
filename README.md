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
\frac{d|e|}{de} = \text{sign}(e) =
\begin{cases}
+1 & e > 0 \\
-1 & e < 0 \\
0 & e = 0
\end{cases}
$$

`0` at `e = 0` isn't a "true" derivative — `|e|` has a sharp corner there — it's the conventional **subgradient** choice. This codebase, however, computes the sign via Rust's built-in [`f64::signum`](https://doc.rust-lang.org/std/primitive.f64.html#method.signum), which returns `+1.0` for `+0.0` (not `0.0`) — so, strictly, a row with `error == 0.0` still nudges the weights as if the error were slightly positive. In practice this almost never matters, since floating-point errors are essentially never exactly `0.0`.

Applying the chain rule through $e_i = w \cdot x_i + b - y_i$:

$$
\frac{\partial MAE}{\partial w_j} = \frac{1}{n} \sum_{i=1}^{n} \text{sign}(e_i) \, x_{i,j}
\qquad
\frac{\partial MAE}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} \text{sign}(e_i)
$$

Unlike MSE's gradient, which shrinks as the error shrinks (proportional to `e`), MAE's gradient has constant magnitude regardless of how close the prediction is — only its *sign* depends on the error. That's why training with MAE tends to be noisier close to the minimum.

## 4. Gradient Descent: The Training Algorithm

Gradient descent is an iterative optimization algorithm. Starting from an initial guess, it repeatedly nudges every parameter a small step against its gradient, walking down the loss surface toward the minimum.

Instead of always computing the gradient over the *entire* dataset before each update, this project splits the `n` rows into consecutive **batches** of `batch_size` rows (the last batch may be smaller if `n` isn't a multiple of `batch_size`), and performs one parameter update **per batch** rather than one per epoch:

- `batch_size = n` — every batch is the whole dataset, so this reduces to plain **batch gradient descent** (one update per epoch, as in the original version of this project).
- `batch_size = 1` — every batch is a single row, i.e. **stochastic gradient descent** (SGD): one update per data point.
- `1 < batch_size < n` — **mini-batch gradient descent**, the general case, trading off the stability of full-batch updates against the speed and noise of per-row updates.

**Algorithm (generalized to `m` features and a `batch_size`):**

1. Initialize $w_1, \dots, w_m = 0$ and $b = 0$.
2. Repeat for a fixed number of `epochs`:
   a. Split the `n` rows into consecutive batches of `batch_size` rows each.
   b. For every batch, in order:
      i. For every row $i$ in the batch, compute the prediction $\hat{y}_i = w \cdot x_i + b$ and the error $e_i = \hat{y}_i - y_i$.
      ii. Accumulate the gradient of the chosen loss with respect to every $w_j$ and $b$, summed only over the rows **in this batch**.
      iii. Divide the accumulated sums by the batch's size, then immediately update every parameter against its gradient, scaled by a **learning rate** $\alpha$, before moving to the next batch:

$$
w_j \leftarrow w_j - \alpha \frac{\partial \, \text{loss}}{\partial w_j} \quad \text{for every } j
\qquad
b \leftarrow b - \alpha \frac{\partial \, \text{loss}}{\partial b}
$$

3. After `epochs` full passes over the data, return the final `(w, b)` as the trained model.

Because each epoch now performs $\lceil n / B \rceil$ updates instead of just one (where $B$ is `batch_size`), smaller batches mean more frequent (but noisier) updates for the same number of epochs.

**Role of $\alpha$ (the learning rate):**
- Too small → convergence is correct but very slow.
- Too large → updates overshoot the minimum and the loss can oscillate or diverge.

**Role of `batch_size`:**
- Smaller batches → more updates per epoch, each based on less data, so gradients are noisier — this noise can help escape shallow local irregularities but makes convergence less smooth.
- Larger batches → gradients are averaged over more rows, giving a more stable and accurate estimate of the true gradient, at the cost of fewer updates per epoch.

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

### `validate_training_input` — shared validation

```rust
fn validate_training_input(x: &Matrix, y: &[f64], alpha: f64, epochs: usize, batch_size: usize) -> Result<(), &'static str>
```

Both `fit_mse` and `fit_mae` validate through this private helper before training starts:

- `x.rows() == y.len()` — one target per sample.
- `x.rows() >= 2` — at least two data points.
- `x.cols() > 0` — at least one feature/column.
- `alpha > 0` and `epochs > 0`.
- `batch_size > 0` — a batch needs at least one row.

Note that `batch_size` is **not** required to divide `x.rows()` evenly, or to be `<= x.rows()` — a `batch_size` larger than the dataset simply results in a single, smaller-than-requested batch (handled by `update_batch`'s `.min(x.rows())`, see below).

### `fit_mse` and `fit_mae` — the two losses as one gradient factor

```rust
pub fn fit_mae(x: &Matrix, y: &[f64], alpha: f64, epochs: usize, batch_size: usize) -> Result<Self, &'static str> {
    Self::fit_with_gradient(x, y, alpha, epochs, batch_size, |error| error.signum())
}

pub fn fit_mse(x: &Matrix, y: &[f64], alpha: f64, epochs: usize, batch_size: usize) -> Result<Self, &'static str> {
    Self::fit_with_gradient(x, y, alpha, epochs, batch_size, |error| 2.0 * error)
}
```

The two losses from Section 2 only ever differ in one place: the factor each per-row error contributes to the gradient (Section 3). `fit_mse` and `fit_mae` are now thin wrappers that just plug in that factor — `2.0 * error` for MSE's $\partial MSE/\partial w_j \propto e_i$, and `error.signum()` for MAE's $\partial MAE/\partial w_j \propto \text{sign}(e_i)$ — and delegate the actual training loop to a single shared implementation, `fit_with_gradient`.

### `fit_with_gradient` — the shared training loop

```rust
fn fit_with_gradient(x: &Matrix, y: &[f64], alpha: f64, epochs: usize, batch_size: usize, gradient_factor: impl Fn(f64) -> f64) -> Result<Self, &'static str>
```

This is the algorithm from Section 4: after validating the input and zero-initializing `weight` (sized `x.cols()`) and `bias`, it loops `epochs` times; on every epoch, it walks `batch_start` from `0` to `x.rows()` in steps of `batch_size` (the last step is clamped with `.min(x.rows())` so the final batch is simply whatever rows remain), calling `update_batch` once per batch — one parameter update per batch, not per epoch.

### `update_batch` — one gradient step over one batch

```rust
fn update_batch(x: &Matrix, y: &[f64], alpha: f64, batch_start: usize, batch_end: usize, weight: &mut [f64], bias: &mut f64, gradient_factor: &impl Fn(f64) -> f64)
```

For the rows `batch_start..batch_end`, it computes each row's prediction (via `prediction_for_row`) and error, applies `gradient_factor(error)` to get the loss-specific contribution (Section 3), and accumulates it into `dw[col]` (multiplied by `x[[row, col]]`) and `db`. After the batch's rows are summed, both are divided by the batch's length and used to update `weight[col]` and `bias` against the gradient, scaled by `alpha` — exactly the update rule in Section 4, but scoped to a single batch instead of the whole dataset.

### `prediction_for_row` — the model's dot product

```rust
fn prediction_for_row(x: &Matrix, row: usize, weight: &[f64], bias: f64) -> f64
```

Computes $\hat{y}_i = w \cdot x_i + b$ from Section 1 for a single row. Shared by `update_batch` (during training) and `predict` (after training), so the prediction logic only exists once.

### `predict` — using the trained model

```rust
pub fn predict(&self, x: &Matrix) -> Result<Vec<f64>, &'static str>
```

Validates that `x.cols() == self.weight.len()` (a model trained on `m` features can't predict on a matrix with a different number of columns), then returns one prediction per row, each computed by `prediction_for_row`.

## 6. Unit Tests

Tests live in [src/lib.rs](src/lib.rs), covering:

- **`predict`** — returns exactly one prediction per matrix row on a hand-computed multi-feature model; returns an error when the input matrix's column count doesn't match the model's weight length.
- **`fit_mse`** — learns a single-feature relation (`y = 2x`) with `batch_size = 5` (i.e. one batch per epoch, equivalent to full-batch gradient descent on that 5-row dataset); learns a two-feature linear relation with `batch_size = 6`; learns a more realistic multi-feature "house price" example with larger-magnitude targets and `batch_size = 8`, verifying both the learned `weight`/`bias` and the resulting predictions land within tolerance.
- **`fit_mae`** — trains on the same single-feature data as `fit_mse` (`batch_size = 5`) and produces a valid model with reasonable predictions.
- **Shared validation** — `fit_mse` is checked to return the correct error for mismatched `x`/`y` row counts, fewer than two rows, `alpha == 0`, `alpha < 0`, `epochs == 0`, and `batch_size == 0`; `fit_mae` is checked to return the same validation error as `fit_mse` for mismatched row counts, confirming both losses share the same `validate_training_input` path.

## 7. Running the Project

This is currently a library crate (no `main.rs` binary) — the way to exercise it is through its test suite:

```bash
cargo test
```
