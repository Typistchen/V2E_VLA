import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from separate_dynamic_static_events import (  # noqa: E402
    ego_geometry,
    fuse_motion_lighting,
    robust_motion_calibration,
)


def _camera_sequence(n_frames=4, height=20, width=30):
    depth = np.ones((n_frames, height, width), np.float32)
    poses = np.zeros((n_frames, 7), np.float64)
    poses[:, 0] = np.arange(n_frames) * 0.1
    poses[:, 3] = 1.0
    intrinsics = np.repeat(
        np.array(
            [[50.0, 0.0, width / 2], [0.0, 50.0, height / 2], [0.0, 0.0, 1.0]],
            np.float64,
        )[None],
        n_frames,
        axis=0,
    )
    return depth, poses, intrinsics


def test_ego_geometry_identity_keeps_depth_and_zero_flow():
    depth, poses, intrinsics = _camera_sequence(n_frames=2)
    poses[1] = poses[0]
    flow, predicted_depth, valid = ego_geometry(
        depth[0], poses[0], poses[1], intrinsics[0]
    )
    assert np.allclose(flow[valid], 0.0, atol=1e-5)
    assert np.allclose(predicted_depth[valid], 1.0, atol=1e-5)


def test_calibration_uses_dominant_geometry_without_semantics():
    depth, poses, intrinsics = _camera_sequence()
    motion = np.zeros((*depth.shape, 2), np.float32)
    # +X camera translation produces -5 px static flow. Isaac convention in
    # this fixture is offset=1, sign=-1, scale=1.25.
    motion[1:, ..., 0] = 4.0
    calibration = robust_motion_calibration(depth, motion, poses, intrinsics)
    assert calibration["offset"] == 1
    assert calibration["sign"] == -1.0
    assert np.isclose(calibration["scale"], 1.25, atol=1e-3)


def test_lighting_without_geometry_is_rejected_from_dynamic():
    shape = (1, 1)
    valid = np.ones(shape, bool)
    static, dynamic, illumination, unknown = fuse_motion_lighting(
        flow_confidence=np.full(shape, 0.9, np.float32),
        depth_confidence=np.zeros(shape, np.float32),
        photo_confidence=np.ones(shape, np.float32),
        chroma_confidence=np.zeros(shape, np.float32),
        persistent=np.zeros(shape, np.float32),
        valid=valid,
    )
    assert illumination.item() > 0.9
    assert dynamic.item() < 0.5
    assert unknown.item() > 0.9
    assert static.item() < 0.1


def test_depth_supported_motion_survives_brightness_change():
    shape = (1, 1)
    _, dynamic, illumination, _ = fuse_motion_lighting(
        flow_confidence=np.full(shape, 0.9, np.float32),
        depth_confidence=np.full(shape, 0.9, np.float32),
        photo_confidence=np.ones(shape, np.float32),
        chroma_confidence=np.zeros(shape, np.float32),
        persistent=np.zeros(shape, np.float32),
        valid=np.ones(shape, bool),
    )
    assert illumination.item() < 0.11
    assert dynamic.item() > 0.8
