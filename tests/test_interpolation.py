import math

import torch

from dvs_gen.warp.interpolation import bidir_warp_gap


def constant_frame(value, height=4, width=5):
    return torch.full((1, height, width, 3), float(value), dtype=torch.float32)


def zero_motion(height=4, width=5):
    return torch.zeros((1, height, width, 2), dtype=torch.float32)


def test_log_blend_is_linear_in_log_luminance():
    a = constant_frame(1.0)
    b = constant_frame(4.0)
    mv = zero_motion()
    depth = torch.ones((1, 4, 5), dtype=torch.float32)

    frames, masks = bidir_warp_gap(
        a, b, mv, mv, 4,
        composite="log_blend",
        depthA=depth,
        depthB=depth,
        return_validity=True,
        valid_margin=0,
    )

    assert len(frames) == len(masks) == 3
    for index, (frame, mask) in enumerate(zip(frames, masks), start=1):
        expected = math.exp(index / 4 * math.log(4.0))
        assert torch.all(mask)
        assert torch.allclose(frame, torch.full_like(frame, expected), atol=2e-5)


def test_depth_disagreement_is_invalid_and_uses_nearer_surface():
    a = constant_frame(1.0)
    b = constant_frame(4.0)
    mv = zero_motion()
    depth_a = torch.ones((1, 4, 5), dtype=torch.float32)
    depth_b = torch.full_like(depth_a, 3.0)

    frames, masks = bidir_warp_gap(
        a, b, mv, mv, 2,
        composite="log_blend",
        depthA=depth_a,
        depthB=depth_b,
        covis_z=True,
        return_validity=True,
        depth_abs_tol=0.01,
        depth_rel_tol=0.0,
        valid_margin=0,
    )

    assert not torch.any(masks[0])
    assert torch.allclose(frames[0], a[0])


def test_motion_disagreement_marks_warp_as_invalid():
    a = constant_frame(1.0)
    b = constant_frame(1.0)
    mv_a = zero_motion()
    mv_b = torch.full_like(mv_a, 0.5)

    _, masks = bidir_warp_gap(
        a, b, mv_a, mv_b, 2,
        composite="log_blend",
        return_validity=True,
        flow_abs_tol=0.1,
        flow_rel_tol=0.0,
        valid_margin=0,
    )

    assert not torch.any(masks[0][1:-1, 1:-1])
