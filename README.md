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
- EVIS photometric/occlusion-aware warp fix: `8ac5418`
- Seed-2 halo evaluation report: `3161a4f`
- DynamicVLA no-warp control support: `9dbea01`

The `demo2` halo fix and its no-warp control are included. See
`isaac-sim-event-camera-plugin/docs/photometric_warp_fix.md` for measured event
counts, temporal-burst metrics, limitations, and the next VLA evaluation step.

See `dynamic-vla/docs/event_camera.md` for setup and generation commands.
