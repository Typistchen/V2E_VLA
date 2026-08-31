import math

import h5py
import numpy as np
import torch

from dvs_gen.dvs.noise import DVSNoiseCfg, DVSNoiseModel
from dvs_gen.dvs.processor import BatchedMultiCamProcessor
from dvs_gen.dvs.recorder import GeneralDVSRecorder


class MemoryRecorder:
    def __init__(self):
        self.batches = []

    def record(self, camera_name, env_ids, xs, ys, ps, t, confidence=None):
        if torch.is_tensor(t):
            times = t.detach().cpu().clone()
        else:
            times = torch.full((xs.numel(),), float(t), dtype=torch.float64)
        quality = (
            torch.ones(xs.numel(), dtype=torch.float32)
            if confidence is None else confidence.detach().cpu().float().clone()
        )
        self.batches.append(
            {
                "camera": camera_name,
                "env": env_ids.detach().cpu().clone(),
                "x": xs.detach().cpu().clone(),
                "y": ys.detach().cpu().clone(),
                "p": ps.detach().cpu().clone(),
                "t": times,
                "q": quality,
            }
        )

    def events(self):
        if not self.batches:
            return {
                "p": torch.empty(0, dtype=torch.int8),
                "t": torch.empty(0, dtype=torch.float64),
            }
        return {
            key: torch.cat([batch[key] for batch in self.batches])
            for key in ("env", "x", "y", "p", "t", "q")
        }


def rgb_from_log(log_intensity):
    intensity = math.exp(log_intensity)
    return torch.full((1, 1, 1, 3), intensity, dtype=torch.float32)


def rgb_pixels_from_logs(log_intensities):
    intensities = torch.exp(torch.tensor(log_intensities, dtype=torch.float32))
    return intensities.reshape(1, 1, -1, 1).expand(-1, -1, -1, 3).clone()


def test_multiple_crossings_keep_residual_and_interpolate_timestamps():
    recorder = MemoryRecorder()
    processor = BatchedMultiCamProcessor(recorder, "cam", threshold=0.15)

    processor(rgb_from_log(0.0), 0.0)
    processor(rgb_from_log(0.46), 1.0)

    events = recorder.events()
    assert events["p"].tolist() == [1, 1, 1]
    np.testing.assert_allclose(
        events["t"].numpy(),
        np.array([0.15, 0.30, 0.45]) / 0.46,
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.allclose(
        processor.ref_log_intensity,
        torch.full_like(processor.ref_log_intensity, 0.45),
        atol=1e-5,
    )

    # The 0.01 residual is retained but remains below threshold, so an unchanged
    # frame must not create another event.
    processor(rgb_from_log(0.46), 2.0)
    assert recorder.events()["p"].numel() == 3


def test_negative_crossings_are_multiple_and_monotonic():
    recorder = MemoryRecorder()
    processor = BatchedMultiCamProcessor(recorder, "cam", threshold=0.15)
    processor(rgb_from_log(0.46), 0.0)
    processor(rgb_from_log(-0.16), 1.0)

    events = recorder.events()
    assert events["p"].tolist() == [-1, -1, -1, -1]
    assert torch.all(events["t"][1:] > events["t"][:-1])
    assert torch.allclose(
        processor.ref_log_intensity,
        torch.full_like(processor.ref_log_intensity, -0.14),
        atol=1e-5,
    )


def test_refractory_filters_crossings_in_timestamp_order():
    recorder = MemoryRecorder()
    noise = DVSNoiseModel(
        DVSNoiseCfg(
            sigma_threshold=0.0,
            refractory_s=1.0e-3,
            leak_rate_hz=0.0,
            shot_rate_hz=0.0,
            hot_pixel_frac=0.0,
            hot_pixel_rate_hz=0.0,
        ),
        threshold=0.1,
    )
    processor = BatchedMultiCamProcessor(recorder, "cam", threshold=0.1, noise=noise)
    processor(rgb_from_log(0.0), 0.0)
    processor(rgb_from_log(0.35), 0.003)

    events = recorder.events()
    assert events["p"].tolist() == [1, 1]
    np.testing.assert_allclose(
        events["t"].numpy(),
        np.array([0.1, 0.3]) / 0.35 * 0.003,
        rtol=1e-5,
        atol=1e-5,
    )


def test_noisy_path_is_globally_timestamp_sorted_across_pixels():
    recorder = MemoryRecorder()
    noise = DVSNoiseModel(
        DVSNoiseCfg(
            sigma_threshold=0.0,
            refractory_s=0.0,
            leak_rate_hz=0.0,
            shot_rate_hz=0.0,
            hot_pixel_frac=0.0,
            hot_pixel_rate_hz=0.0,
        ),
        threshold=0.1,
    )
    processor = BatchedMultiCamProcessor(recorder, "cam", threshold=0.1, noise=noise)
    processor(rgb_pixels_from_logs([0.0, 0.0]), 0.0)
    processor(rgb_pixels_from_logs([0.11, 0.35]), 0.003)

    events = recorder.events()
    assert events["p"].numel() == 4
    assert torch.all(events["t"][1:] >= events["t"][:-1])


def test_invalid_warp_pixel_is_suppressed_and_reanchored_on_recovery():
    recorder = MemoryRecorder()
    processor = BatchedMultiCamProcessor(recorder, "cam", threshold=0.15)
    valid = torch.ones((1, 1, 1), dtype=torch.bool)
    invalid = torch.zeros_like(valid)

    processor(rgb_from_log(0.0), 0.0, valid)
    processor(rgb_from_log(0.46), 1.0, invalid)
    assert recorder.events()["p"].numel() == 0

    # Recovery must establish a fresh reference without emitting the unknown
    # accumulated change from the invalid interval.
    processor(rgb_from_log(0.46), 2.0, valid)
    assert recorder.events()["p"].numel() == 0

    processor(rgb_from_log(0.62), 3.0, valid)
    assert recorder.events()["p"].tolist() == [1]


def test_recorder_writes_per_event_timestamps(tmp_path):
    recorder = GeneralDVSRecorder(str(tmp_path), compression=None)
    recorder.record(
        "DVS/cam",
        torch.tensor([0, 0, 0]),
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5, 6]),
        torch.tensor([1, 1, -1], dtype=torch.int8),
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
    )
    recorder.flush_episode(0, 7)

    with h5py.File(tmp_path / "env0_ep7.h5", "r") as stream:
        np.testing.assert_allclose(stream["DVS/cam/t"][:], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(stream["DVS/cam/q"][:], [1.0, 1.0, 1.0])


def test_soft_confidence_is_attached_without_suppressing_events():
    recorder = MemoryRecorder()
    processor = BatchedMultiCamProcessor(recorder, "cam", threshold=0.15)

    processor(rgb_from_log(0.0), 0.0, confidence=torch.tensor([[[0.8]]]))
    processor(rgb_from_log(0.46), 1.0, confidence=torch.tensor([[[0.3]]]))

    events = recorder.events()
    assert events["p"].tolist() == [1, 1, 1]
    assert torch.allclose(events["q"], torch.full((3,), 0.3))
