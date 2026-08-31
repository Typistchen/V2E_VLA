# V2E_VLA

Event-camera simulation for DynamicVLA, combining the EVIS Isaac Sim plugin
with the DOM/DynamicVLA simulation pipeline.

## Repository layout

- `isaac-sim-event-camera-plugin/`: EVIS event generation, multi-threshold
  event model, noise/refractory handling, HDF5 recording, and visualization.
- `dynamic-vla/`: DynamicVLA DOM simulation integration and usage docs.

Both directories retain their original Git histories through subtree imports.
Generated datasets and videos are intentionally not tracked.

## Current checkpoints

- EVIS core multi-threshold event model: `a6eec8b`
- EVIS Isaac/H.264 compatibility: `50a1a7c`
- EVIS event-video tooling: `3967743`
- DynamicVLA EVIS integration: `df4526b`

The photometric/occlusion-aware warp fix discussed for the `demo2` halo
artifact is not included yet and should be developed as a separate commit.

See `dynamic-vla/docs/event_camera.md` for setup and generation commands.
