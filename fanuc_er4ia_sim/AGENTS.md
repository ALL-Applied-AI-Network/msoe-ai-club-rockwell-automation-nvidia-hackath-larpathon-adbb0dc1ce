# Agent setup and development instructions

## Objective

Set up this starter on the participant's machine and verify the rotated-cube
stacking task. Read `README.md`, `SETUP.md` and `docs/MODEL_NOTES.md` before editing.
Read `docs/CALIBRATION_V05.md` before changing geometry or camera calibration.
User instructions take precedence over this file.

## Establish the environment first

1. Inspect the current directory, host Python version, GPU availability and any
   existing Isaac/Cosmos installations. Use the participant's actual paths.
2. Run `./bootstrap.sh` to prepare the host environment, generate materials and
   run the 33 host tests. Reuse an existing working `.venv`.
3. Configure `ISAAC` and `COSMOS`. Set `COSMOS3_NANO_PATH` to a real model snapshot.
   Set `COSMOS3_ROOT` only if the framework package is not already installed.
4. Run the dependency/model checks in `SETUP.md`. Do not execute placeholder
   `/path/to/...` values. If software or weights are missing, report the exact
   dependency and continue any independent host setup/checks.
5. Build the scene, then execute the full pipeline in an empty `runs/` directory.

The host, Isaac and Cosmos Python interpreters have different roles. Never
substitute the host interpreter for a simulator script or model inference.
Do not copy a development machine's absolute paths into code.

## Preserve task semantics

- Six cubes: two yellow, two blue, two red. Stack in that bottom-to-top order.
- Use the table camera for perception. Five cubes start rotated.
- Gripper actuator: SMC MHZL2-16D2, 6 mm/jaw, 12 mm straight-face opening.
- Align the wrist to the observed cube pose, then align the cube for placement.
- The image front end supplies quantitative pose to Cosmos. Describe this as
  hybrid perception, not independent VLM pose estimation or arbitrary 6D pose.
- Ground-truth files and tracked simulator object poses are for evaluation only.
  Never use them to fix detections or supply grasp commands.
- Preserve confidence, inventory, pose, pickup and support gates. Diagnose a
  failed stage rather than weakening thresholds to make the run pass.

## Validation and reporting

Run the host tests after code changes:

```bash
.venv/bin/python -m unittest test_sort_planning test_cube_perception test_cube_pose test_runtime_config
```

For geometry, physics, camera or motion changes, also capture a fresh scene and
validate physical execution. Inspect `execution_report.json`: require
`task_complete: true`, `verified_pickups: 6` and `verified_placements: 6`.
Isaac can return exit code zero after an exception; a successful process exit
alone is insufficient. Inspect the report and logs.

Do not claim GPU/model/physics validation from host tests alone. If a dependency
blocks a check, state what was actually run and what remains unverified.

## Package boundary

Only runtime code, instructions, tests and the five geometry assets are shipped.
Procedural textures, captures and videos are generated locally; photo maps,
calibration source photos, old experiments and recorded runs are absent.
`package_hackathon.py` uses an explicit allowlist and creates a SHA-256 manifest.
Keep starter packages free of media, caches, credentials and environments.
