from functools import partial
import numpy as np
from typing import Tuple


def create_transforms(source_grid_scaled, deg=11, mesh_weight_map=None):

    import jax
    import jax.numpy as jnp
    from jax.tree_util import register_pytree_node_class

    @jax.jit
    def interp1d(x, xp, fp):  #, left=None, right=None):
        i = jnp.clip(
            jnp.searchsorted(
                xp,
                x,
                side='right',
                method='scan_unrolled'
            ),
            1,
            len(xp) - 1
        )
        df = fp[i] - fp[i - 1]
        dx = xp[i] - xp[i - 1]
        delta = x - xp[i - 1]
        eps = jnp.finfo(xp.dtype).eps
        epsilon = jnp.nextafter(eps, jnp.inf) - eps

        dx0 = jax.lax.abs(dx) <= epsilon  # Prevent NaN gradients when `dx` is small.
        f = jnp.where(
            dx0,
            fp[i - 1],
            fp[i - 1] + (delta / jnp.where(dx0, 1, dx)) * df
        )

        return f

    @jax.custom_jvp
    def spline_invert(ip, x):
        # use a custom jvp because we are using cached values to get the spline faster
        # and this would not easily give the grad with respect to the poly coefs as written
        k_right = jnp.digitize(x, ip.x_low_res, method='scan_unrolled')
        k_left = k_right - 1

        # jax's default out-of-bound index gives
        # correct result for point on the right most
        # edge of interpolation, no need to do anything
        # special for the boundary
        t = (x - ip.x_low_res[k_left]) / ip.delta_x[k_left]
        t2 = t ** 2
        t3 = t ** 3
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        return ip.y_low_res[k_left] * h00 + ip.y_low_res[k_right] * h01 + (
                    ip.dy_low_res[k_left] * h10 + ip.dy_low_res[k_right] * h11) * ip.delta_x[k_left]

    @spline_invert.defjvp
    def invert_poly_jvp(primals, tangents):
        # because this is the inverse of a polynomial it's
        # gradient can be written in terms of the gradient
        # of the polynomial evaluated at the output
        # this is easy to write down and avoids needing
        # to grad through the cubic spline inversion
        ip, x = primals
        ip_dot, x_dot = tangents
        primal_out = spline_invert(ip, x)
        d_dx = 1 / jnp.polyval(ip.dcoefs, primal_out)
        d_dcoefs = -jnp.vander(jnp.atleast_1d(primal_out), N=ip.coefs.shape[0])
        tangent_out = ((ip_dot.coefs * d_dcoefs).sum() + x_dot) * d_dx
        return primal_out, tangent_out

    @register_pytree_node_class
    class InvertPolySpline:
        @staticmethod
        def v_polyder(c):
            return jax.vmap(
                jnp.polyder,
                in_axes=1,
                out_axes=1
            )(c)

        @staticmethod
        def v_polyval(c, x):
            return jax.vmap(
                jnp.polyval,
                in_axes=(1, 1),
                out_axes=(1)
            )(c, x)

        def __init__(self, coefs, lower_bound, upper_bound, low_res=150):
            # coefs Nx2
            # lower_bound 1x2
            # upper_bound 1x2
            # low_res int

            # polynomial to inverse
            self.coefs = coefs

            # get 1st derivative of polynomial
            self.dcoefs = InvertPolySpline.v_polyder(self.coefs)

            # The bounds of the CDF
            # below will always be 0
            # above will always be 1
            self.lower_bound = jnp.atleast_2d(lower_bound)
            self.upper_bound = jnp.atleast_2d(upper_bound)

            # low resolution grid of nodes for spline approx to the inverse function
            # cubic spline needs the function, derivative, and delta_x at each node
            self.low_res = low_res
            y_low_res = jnp.linspace(0, 1, low_res)
            self.y_low_res = jnp.stack([y_low_res, y_low_res], axis=1)
            self.x_low_res = InvertPolySpline.v_polyval(self.coefs, self.y_low_res)
            self.dy_low_res = 1 / InvertPolySpline.v_polyval(self.dcoefs, self.y_low_res)
            self.delta_x = jnp.diff(self.x_low_res, axis=0)

        def __repr__(self):
            return f'InvertPoly(coefs={self.coefs}, lower_bound={self.lower_bound}, upper_bound={self.upper_bound})'

        def tree_flatten(self):
            children = (
                self.coefs,
                self.dcoefs,
                self.y_low_res,
                self.x_low_res,
                self.dy_low_res,
                self.delta_x,
                self.lower_bound,
                self.upper_bound
            )
            aux_data = (self.low_res,)
            return (children, aux_data)

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            # return cls(*(children + aux_data))
            obj = object.__new__(InvertPolySpline)
            obj.coefs = children[0]
            obj.dcoefs = children[1]
            obj.y_low_res = children[2]
            obj.x_low_res = children[3]
            obj.dy_low_res = children[4]
            obj.delta_x = children[5]
            obj.lower_bound = children[6]
            obj.upper_bound = children[7]
            obj.low_res = aux_data[0]
            return obj

        def fwd_transform(self, x):
            y = jax.vmap(
                spline_invert,
                in_axes=(1, 1),
                out_axes=(1)
            )(self, x)
            y = jnp.where(x <= self.lower_bound, 0.0, y)
            y = jnp.where(x >= self.upper_bound, 1.0, y)
            return jnp.clip(y, 0.0, 1.0)

        def rev_transform(self, y):
            return InvertPolySpline.v_polyval(self.coefs, y)

    v_polyfit = jax.vmap(jnp.polyfit, in_axes=(1, 1, None, None, None, 1), out_axes=(1))
    v_gradient = jax.vmap(jnp.gradient, in_axes=(1, 1), out_axes=1)

    # inv_poly is a pytree, it can be returned from `jit` without issue :D
    @partial(jax.jit, static_argnames=('deg'))
    def create_transforms_spline(traced_points, deg=11, mesh_weight_map=None):

        N = traced_points.shape[0]  # // 2
        if mesh_weight_map is None:
            t = jnp.arange(1, N + 1) / (N + 1)
            t = jnp.stack([t, t], axis=1)
            sort_points = jnp.sort(traced_points, axis=0)  # [::2]
        else:
            sdx = jnp.argsort(traced_points, axis=0)
            sort_points = jnp.take_along_axis(traced_points, sdx, axis=0)
            t = jnp.stack([mesh_weight_map, mesh_weight_map], axis=1)
            t = jnp.take_along_axis(t, sdx, axis=0)
            t = jnp.cumsum(t, axis=0)

        # The CDF estimation needs to be a smooth function to avoid noise caused by
        # using a sub-set of traced points
        #
        # A polynomial is fit to the *inverse* CDF, this polynomial is inverted
        # numerically to get the smooth CDF function.
        #
        # The polynomial is fit to 'y' points at the Chebyshev nodes to avoid the
        # Runge phenomenon and to estimate the gradient of the CDF
        #
        # The gradient of the CDF is use as the weights for the polynomial fit
        # (e.g. where the CDF changes rapidly the weight is higher).  This
        # helps prevent overfitting for high degree polynomials and helps keep
        # log degree polynomials monotonic.

        # Use 3x more Chebyshev nodes than the degree being fit
        cheb_deg = 3 * deg
        # calculate nodes and interpolated values at the nodes
        cheb_nodes = ((jnp.cos((2 * jnp.arange(cheb_deg) + 1) * jnp.pi / (2 * cheb_deg))[::-1]) + 1) / 2
        cy = jnp.stack([cheb_nodes, cheb_nodes], axis=1)
        cx = jax.vmap(interp1d, in_axes=(None, 1, 1), out_axes=1)(cheb_nodes, t, sort_points)

        # fit the polynomial with weights
        w = v_gradient(cy, cx)
        coefs = v_polyfit(cy, cx, deg, None, False, w)

        # invert the polynomial with custom class
        inv_poly = InvertPolySpline(coefs, sort_points[0], sort_points[-1], low_res=20 * deg)
        # return sort_points and t for plotting below, in production we just need the transforms
        return inv_poly, sort_points, t

    ips, sort_points, t = create_transforms_spline(source_grid_scaled, deg=deg, mesh_weight_map=mesh_weight_map)

    transform = jax.jit(ips.fwd_transform)
    inv_transform = jax.jit(ips.rev_transform)

    return transform, inv_transform