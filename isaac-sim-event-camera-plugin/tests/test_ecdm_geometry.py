import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_ego_dynamic_events.py"
SPEC = importlib.util.spec_from_file_location("extract_ego_dynamic_events", SCRIPT)
ECDM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ECDM)


def _camera_inputs():
    depth = np.full((20, 30), 2.0, np.float32)
    pose = np.asarray([0, 0, 0, 1, 0, 0, 0], np.float64)
    intrinsics = np.asarray(
        [[100, 0, 15], [0, 100, 10], [0, 0, 1]], np.float64
    )
    return depth, pose, intrinsics


def test_identity_camera_motion_has_zero_ego_flow():
    depth, pose, intrinsics = _camera_inputs()
    flow, valid = ECDM.ego_flow(depth, pose, pose, intrinsics)
    # Floating-point projection may reject the outermost zero-coordinate row.
    assert np.mean(valid) > 0.95
    np.testing.assert_allclose(flow[valid], 0.0, atol=1e-5)


def test_camera_translation_predicts_static_background_flow():
    depth, pose0, intrinsics = _camera_inputs()
    pose1 = pose0.copy()
    pose1[0] = 0.1
    flow, valid = ECDM.ego_flow(depth, pose0, pose1, intrinsics)
    # Camera moves +X, so a static point shifts left by fx * tx / depth = 5 px.
    np.testing.assert_allclose(flow[..., 0][valid], -5.0, atol=1e-5)
    np.testing.assert_allclose(flow[..., 1][valid], 0.0, atol=1e-5)
