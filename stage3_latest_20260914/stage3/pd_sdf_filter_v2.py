"""New optional guide gate; leave the frozen original filter/source untouched.

Isaac Gym reset may update root state before rigid-body tensors are refreshed.
Do not query a scene or Jacobian on mismatched states. Pass through the trained
action until physics supplies coherent root and body transforms; never move it.
"""
import torch
from pd_sdf_filter import PDSDFFilter


class PDSDFFilterV2(PDSDFFilter):
    def __call__(self, action):
        if action.shape != (1, 28) or not bool(torch.isfinite(action).all()) or float(action.abs().max()) > 1.000001:
            raise ValueError('Finite bounded native PD action required')
        task = self.task
        root = task._humanoid_root_states[0]
        body_position = task._rigid_body_pos[0, 0]
        body_rotation = task._rigid_body_rot[0, 0]
        if not bool(torch.isfinite(root).all() and torch.isfinite(body_position).all() and torch.isfinite(body_rotation).all()):
            raise ValueError('Nonfinite observed physical state')
        position_error = float((body_position-root[:3]).abs().max())
        rotation_error = min(float((body_rotation-root[3:7]).abs().max()),
                             float((body_rotation+root[3:7]).abs().max()))
        if position_error > 1e-4 or rotation_error > 1e-4:
            self.trace.append(dict(applied=False, reason='stale_rigid_body_root_state',
                rigid_root_position_max_error_m=position_error,
                rigid_root_quaternion_sign_invariant_max_error=rotation_error,
                state_consistency_position_tolerance_m=1e-4,
                state_consistency_quaternion_tolerance=1e-4,
                jacobian_velocity_rmse=None, jacobian_velocity_check_performed=False,
                max_normalized_action_delta=0.0, surface_sample_count=len(self.ids),
                actual_physical_transition=True, root_state_modified=False,
                learned_policy_weights_modified=False, full_surface_guarantee=False))
            return action
        result = super().__call__(action)
        self.trace[-1].update(rigid_root_position_max_error_m=position_error,
            rigid_root_quaternion_sign_invariant_max_error=rotation_error,
            state_consistency_position_tolerance_m=1e-4,
            state_consistency_quaternion_tolerance=1e-4,
            jacobian_velocity_check_performed=True)
        return result
