import numpy as np
import openmdao.api as om
import pandas as pd
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint


class LiftDragRatio(om.ExplicitComponent):
    """
    Calculate lift-to-drag ratio (L/D) based on section forces.
    This is for one given lifting surface and provides an important
    aerodynamic efficiency metric that can be used as a constraint
    or objective in optimization problems.

    Parameters
    ----------
    sec_forces[nx-1, ny-1, 3] : numpy array
        Contains the sectional forces acting on each panel.
    alpha : float
        Angle of attack in degrees.
    beta : float
        Sideslip angle in degrees.

    Returns
    -------
    L_over_D : float
        Lift-to-drag ratio for the lifting surface.
    L : float
        Total induced lift force for the lifting surface (for reference).
    D : float
        Total induced drag force for the lifting surface (for reference).
    """

    def initialize(self):
        self.options.declare("surface", types=dict)

    def setup(self):
        self.surface = surface = self.options["surface"]

        self.nx = nx = surface["mesh"].shape[0]
        self.ny = ny = surface["mesh"].shape[1]
        self.num_panels = (nx - 1) * (ny - 1)

        # Inputs
        self.add_input("sec_forces", val=np.ones((nx - 1, ny - 1, 3)), units="N", tags=["mphys_coupling"])
        self.add_input("alpha", val=3.0, units="deg", tags=["mphys_input"])
        self.add_input("beta", val=0.0, units="deg", tags=["mphys_input"])
        
        # Outputs
        self.add_output("L_over_D", val=10.0, units=None)  # Dimensionless ratio
        self.add_output("L", val=0.0, units="N")  # For reference/debugging
        self.add_output("D", val=0.0, units="N")  # For reference/debugging

        # Declare partial derivatives
        self.declare_partials(["L_over_D", "L", "D"], ["sec_forces", "alpha"])
        self.declare_partials(["L_over_D", "D"], "beta")

        # Small number to prevent division by zero
        self.eps = 1e-12

    def compute(self, inputs, outputs):
        """Compute lift, drag, and lift-to-drag ratio."""
        alpha = inputs["alpha"] * np.pi / 180.0
        beta = inputs["beta"] * np.pi / 180.0
        forces = inputs["sec_forces"].reshape(-1, 3)

        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)

        # Compute the induced lift force
        L = np.sum(-forces[:, 0] * sina + forces[:, 2] * cosa)

        # Compute the induced drag force
        D = np.sum(forces[:, 0] * cosa * cosb - forces[:, 1] * sinb + forces[:, 2] * sina * cosb)

        # Apply symmetry factor 
        if self.surface["symmetry"]:
            L *= 2.0
            D *= 2.0

        # Store outputs
        outputs["L"] = L
        outputs["D"] = D

        # Compute L/D ratio with protection against division by zero
        if np.abs(D) > self.eps:
            outputs["L_over_D"] = L / D
        else:
            outputs["L_over_D"] = 0.0

    def compute_partials(self, inputs, partials):
        """Compute analytical derivatives for lift-to-drag ratio."""
        
        # Convert inputs
        p180 = np.pi / 180.0
        alpha = inputs["alpha"][0] * p180
        beta = inputs["beta"][0] * p180

        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)

        forces = inputs["sec_forces"]

        # Symmetry factor
        if self.surface["symmetry"]:
            symmetry_factor = 2.0
        else:
            symmetry_factor = 1.0

        # Compute L and D
        L = symmetry_factor * np.sum(-forces[:, :, 0] * sina + forces[:, :, 2] * cosa)
        D = symmetry_factor * np.sum(forces[:, :, 0] * cosa * cosb - forces[:, :, 1] * sinb + forces[:, :, 2] * sina * cosb)

        # Derivatives of L and D with respect to sec_forces
        dL_dforces = np.array([-sina, 0, cosa])
        dD_dforces = np.array([cosa * cosb, -sinb, sina * cosb])

        partials["L", "sec_forces"] = np.atleast_2d(np.tile(dL_dforces, self.num_panels)) * symmetry_factor
        partials["D", "sec_forces"] = np.atleast_2d(np.tile(dD_dforces, self.num_panels)) * symmetry_factor

        # Derivatives of L and D with respect to alpha
        dL_dalpha = p180 * symmetry_factor * np.sum(-forces[:, :, 0] * cosa - forces[:, :, 2] * sina)
        dD_dalpha = p180 * symmetry_factor * np.sum(-forces[:, :, 0] * sina * cosb + forces[:, :, 2] * cosa * cosb)

        partials["L", "alpha"] = dL_dalpha
        partials["D", "alpha"] = dD_dalpha

        # Derivative of D with respect to beta
        dD_dbeta = p180 * symmetry_factor * np.sum(
            -forces[:, :, 0] * cosa * sinb - forces[:, :, 1] * cosb - forces[:, :, 2] * sina * sinb
        )
        partials["D", "beta"] = dD_dbeta

        # Derivatives of L/D ratio using quotient rule: d(L/D)/dx = (dL/dx * D - L * dD/dx) / D^2
        
        # L/D derivatives with respect to sec_forces
        dLD_dforces = np.zeros((1, self.num_panels * 3))
        for i in range(self.num_panels):
            for j in range(3):
                idx = i * 3 + j
                dL_df = dL_dforces[j] * symmetry_factor
                dD_df = dD_dforces[j] * symmetry_factor
                
                if np.abs(D) > self.eps:
                    dLD_dforces[0, idx] = (dL_df * D - L * dD_df) / (D * D)
                else:
                    dLD_dforces[0, idx] = 0.0
        
        partials["L_over_D", "sec_forces"] = dLD_dforces

        # L/D derivative with respect to alpha
        if np.abs(D) > self.eps:
            partials["L_over_D", "alpha"] = (dL_dalpha * D - L * dD_dalpha) / (D * D)
        else:
            partials["L_over_D", "alpha"] = 0.0

        # L/D derivative with respect to beta
        if np.abs(D) > self.eps:
            partials["L_over_D", "beta"] = (-L * dD_dbeta) / (D * D)
        else:
            partials["L_over_D", "beta"] = 0.0