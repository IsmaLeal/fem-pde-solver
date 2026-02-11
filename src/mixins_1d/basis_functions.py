import numpy as np
from typing import Union


class BasisFunctionMixin:
    def phi_linear(
            self,
            x: Union[float, np.ndarray],
            x_i: float
    ) -> Union[float, np.ndarray]:
        '''
        Compute the linear basis function value for a given node in 1D finite elements.

        This function evaluates the linear basis function associated with node 'x_i' at the
        evaluation points 'x'. It supports both scalar ('float') and vectorised ('ndarray') inputs.

        Parameters
        ----------
        x : float or ndarray
            Evaluation point (if scalar) or array of evaluation points.
        x_i : float
            Node for which the basis function is computed.

        Returns
        -------
        phi_values : float or ndarray
            The value of the basis function at 'x' (if scalar) or an array of basis function
            values at the corresponding points in 'x' (if vectorised).

        Notes
        -----
        - The basis function is piecewise linear and depends on the neighbouring node distances,
        stored in 'self.dxs'.
        - 'self.node_to_idx' is a mapping of node coordinates to their respective indices in 'self.dxs'.
        '''
        i = int(self.node_to_idx[x_i])

        # Quadrature function 'quad()' passes floats as inputs
        if type(x) == float:
            if x > x_i:
                phi_values = np.maximum(1 - (x - x_i) / self.dxs[i], 0)
            else:
                phi_values = np.maximum(1 - (x_i - x) / self.dxs[i - 1], 0)

        # Array input
        else:
            left_mask = x <= x_i
            right_mask = x > x_i

            phi_values = np.zeros_like(x)
            phi_values[left_mask] = np.maximum(1 - (x_i - x[left_mask]) / self.dxs[i - 1], 0)
            if i < len(self.dxs):
                phi_values[right_mask] = np.maximum(1 - (x[right_mask] - x_i) / self.dxs[i], 0)
        return phi_values



    def phi_linear_derivative(
            self,
            x: float,
            x_i: float
    ) -> float:
        '''
        Vectorized derivative of the linear basis function for FEM. Accepts both arrays and floats as inputs

        Parameters
        ----------
        x : float
            Evaluation point
        x_i : float
            Node for which the basis function derivative is computed.

        Returns
        -------
        gradient : float
            Derivative of the x_i basis functions.
        '''
        node_idx = int(self.node_to_idx[x_i])
        if self.x_nodes[node_idx - 1] <= x <= x_i:
            dx_left = self.x_nodes[node_idx] - self.x_nodes[node_idx - 1]
            gradient = 1 / dx_left if x_i != self.x_nodes[0] else 0
            return gradient
        elif x_i < x <= self.x_nodes[node_idx + 1]:
            dx_right = self.x_nodes[node_idx + 1] - self.x_nodes[node_idx]
            gradient = -1 / dx_right if x_i != self.x_nodes[-1] else 0
            return gradient
        else:
            return 0
