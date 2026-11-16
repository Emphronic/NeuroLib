//! B-spline KAN layer: base path (SiLU + linear) + spline path.
//!
//! The spline basis is a real B-spline basis computed by the Cox-de Boor
//! recursion, as in the Kolmogorov-Arnold Networks paper (Liu et al., 2024) and
//! the reference `pykan` implementation. Correctness of the recursion
//! (partition of unity, output shape) is verified numerically. Repeating each
//! input `n_coeff` times instead would reduce the spline path to a linear layer
//! and remove the trainable nonlinear edge function that defines KAN.

use burn::module::{Module, Param};
use burn::tensor::Device;
use burn::tensor::activation::silu;
use burn::tensor::{Distribution, Tensor};

/// Configuration [`KANLinear`].
#[derive(Debug, Clone, Copy)]
pub struct KANLinearConfig {
    /// Input size.
    pub in_features: usize,
    /// Output size.
    pub out_features: usize,
    /// The number of internal nodes of the B-spline (without taking into account expansion by order).
    pub grid_size: usize,
    /// Spline order (3 = cubic, as in the original KAN article).
    pub spline_order: usize,
    /// The grid range `[lo, hi]` on which splines are defined.
    pub grid_range: [f32; 2],
    /// Initialization scale of the base (linear) path.
    pub scale_base: f32,
    /// Spline path initialization scale.
    pub scale_spline: f32,
}

impl Default for KANLinearConfig {
    fn default() -> Self {
        Self {
            in_features: 1,
            out_features: 1,
            grid_size: 5,
            spline_order: 3,
            grid_range: [-2.0, 2.0],
            scale_base: 0.5,
            scale_spline: 0.1,
        }
    }
}

impl KANLinearConfig {
    /// New configuration with mandatory dimensions.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            ..Default::default()
        }
    }

    /// Set the grid size.
    pub fn with_grid_size(mut self, grid_size: usize) -> Self {
        self.grid_size = grid_size;
        self
    }

    /// Set the spline order.
    pub fn with_spline_order(mut self, order: usize) -> Self {
        self.spline_order = order;
        self
    }

    /// Set the grid range.
    pub fn with_grid_range(mut self, range: [f32; 2]) -> Self {
        self.grid_range = range;
        self
    }

    /// Create a layer.
    pub fn init(&self, device: &Device) -> KANLinear {
        KANLinear::new(
            self.in_features,
            self.out_features,
            self.grid_size,
            self.spline_order,
            self.grid_range,
            self.scale_base,
            self.scale_spline,
            device,
        )
    }
}

/// KAN-linear layer: `y = silu(x)·W_base + spline(x)·W_spline`.
///
/// Spline part: each input value `x_i` is expanded into a basis
/// from `grid_size + spline_order` B-splines (the sum of which is equal to 1 at any
/// point of the domain - the partition of unity property), then this decomposition
/// multiplied by the trained weights `spline_weight`. So each edge "input →
/// output" receives its trainable piecewise polynomial function, rather than
/// just a number, like in a regular linear layer.
#[derive(Module, Debug)]
pub struct KANLinear {
    /// Base weight `[out_features, in_features]`.
    pub base_weight: Param<Tensor<2>>,
    /// Spline weight `[out_features, in_features * n_coeff]`.
    pub spline_weight: Param<Tensor<2>>,
    in_features: usize,
    out_features: usize,
    grid_size: usize,
    spline_order: usize,
    n_coeff: usize,
    /// Extended vector of mesh nodes (knots), fixed, not trained.
    /// Length `grid_size + 2*spline_order + 1`.
    knots: Vec<f32>,
}

impl KANLinear {
    /// Create a layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_features: usize,
        out_features: usize,
        grid_size: usize,
        spline_order: usize,
        grid_range: [f32; 2],
        scale_base: f32,
        scale_spline: f32,
        device: &Device,
    ) -> Self {
        assert!(grid_size >= 1, "grid_size must be >= 1");
        assert!(
            grid_range[1] > grid_range[0],
            "grid_range: hi must be greater than lo"
        );

        let n_coeff = grid_size + spline_order;
        let knots = build_knots(grid_size, spline_order, grid_range);

        let base_weight = Param::from_tensor(Tensor::random(
            [out_features, in_features],
            Distribution::Uniform(-scale_base as f64, scale_base as f64),
            device,
        ));

        // The initialization scale of spline weights is divided by n_coeff: without
        // In this case, the dispersion of the spline path output would increase proportionally
        // number of basis functions (the sum of n_coeff terms with
        // comparable amplitude), and the spline path would dominate
        // base-path already at the start, and not after training.
        let sc = (scale_spline as f64) / (n_coeff as f64).sqrt();
        let spline_weight = Param::from_tensor(Tensor::random(
            [out_features, in_features * n_coeff],
            Distribution::Uniform(-sc, sc),
            device,
        ));

        Self {
            base_weight,
            spline_weight,
            in_features,
            out_features,
            grid_size,
            spline_order,
            n_coeff,
            knots,
        }
    }

    /// True B-spline basis (Cox–de Boer recursion).
    ///
    /// Input `x`: `[batch, in_features]`, values outside `grid_range` are clamped
    /// to the boundary - the equivalent of a "clamped" spline, standard practice for
    /// input-dependent bases for inference outside the learning range.
    ///
    /// Output: `[batch, in_features * n_coeff]` - the same layout that expects
    /// `spline_weight`.
    fn basis(&self, x: Tensor<2>) -> Tensor<2> {
        let [batch, in_features] = x.dims();

        let lo = self.knots[self.spline_order];
        let hi = self.knots[self.knots.len() - 1 - self.spline_order];
        let x = x.clamp(lo, hi);

        // [batch, in_features, 1]
        let xe = x.unsqueeze_dim::<3>(2);

        // Order 0: indicator functions of half-intervals [k_i, k_{i+1}).
        // bases[j] lives on nodes knots[j]..knots[j+1].
        let n_knots = self.knots.len();
        let mut bases: Vec<Tensor<3>> = Vec::with_capacity(n_knots - 1);
        for j in 0..n_knots - 1 {
            let k_lo = self.knots[j];
            let k_hi = self.knots[j + 1];
            let ge = xe.clone().greater_equal_elem(k_lo).float();
            // The last interval is closed on the right so that the domain boundary
            // fell into the basis (otherwise x == hi would give zero everywhere).
            let lt = if j + 2 == n_knots {
                xe.clone().lower_equal_elem(k_hi).float()
            } else {
                xe.clone().lower_elem(k_hi).float()
            };
            bases.push(ge.mul(lt));
        }

        // Cox-de Boer recursion increases the order from 0 to spline_order.
        // B_{j,k}(x) = (x - t_j)/(t_{j+k} - t_j) * B_{j,k-1}(x)
        //            + (t_{j+k+1} - x)/(t_{j+k+1} - t_{j+1}) * B_{j+1,k-1}(x)
        for k in 1..=self.spline_order {
            let mut next = Vec::with_capacity(bases.len() - 1);
            for j in 0..bases.len() - 1 {
                let t_j = self.knots[j];
                let t_jk = self.knots[j + k];
                let t_j1 = self.knots[j + 1];
                let t_jk1 = self.knots[j + k + 1];

                let denom_l = t_jk - t_j;
                let denom_r = t_jk1 - t_j1;

                // The nodes are built evenly (see build_knots), so
                // The denominators are strictly positive and there is no division by zero.
                let left = xe
                    .clone()
                    .sub_scalar(t_j)
                    .div_scalar(denom_l)
                    .mul(bases[j].clone());
                let right = xe
                    .clone()
                    .mul_scalar(-1.0)
                    .add_scalar(t_jk1)
                    .div_scalar(denom_r)
                    .mul(bases[j + 1].clone());

                next.push(left + right);
            }
            bases = next;
        }

        debug_assert_eq!(bases.len(), self.n_coeff);

        // [batch, in_features, n_coeff] -> [batch, in_features * n_coeff]
        let stacked = Tensor::stack::<4>(bases, 2); // [b, in, n_coeff, 1]
        stacked.reshape([batch, in_features * self.n_coeff])
    }

    /// Direct passage.
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let base_out = silu(x.clone()).matmul(self.base_weight.val().transpose());

        let basis = self.basis(x);
        let spline_out = basis.matmul(self.spline_weight.val().transpose());

        base_out + spline_out
    }
}

/// Constructs an extended uniform vector of nodes of length
/// `grid_size + 2*order + 1` on the range `[lo, hi]`.
///
/// Nodes extended by `order` grid steps per boundary - standard
/// technique for clamped B-splines: without expansion of basis functions at the edges
/// domains would be only partially defined and would not add up to 1.
fn build_knots(grid_size: usize, order: usize, [lo, hi]: [f32; 2]) -> Vec<f32> {
    let h = (hi - lo) / grid_size as f32;
    (0..grid_size + 2 * order + 1)
        .map(|i| lo + h * (i as f32 - order as f32))
        .collect()
}
