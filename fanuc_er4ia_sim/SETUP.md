# Setup checklist for people and coding agents

Run commands from the extracted `fanuc_er4ia_sim/` directory. Replace all
`/path/to/...` placeholders with actual machine paths. The starter has no media
or saved run to replay. It was validated on Linux with Isaac Sim 6.0 and an RTX
A6000; Windows and smaller GPUs have not been benchmarked.

## 1. Host setup

Prerequisites: Python with `venv` and pip. Host Python 3.12 was used for validation.

```bash
./bootstrap.sh
```

This creates/reuses `.venv`, installs `requirements.txt`, generates procedural
metal/environment maps in `textures/`, and runs 33 tests. No reference images
are needed. Detailed finger meshes use silver materials without photo maps.
These generated maps are local outputs, not bundled image files.

## 2. Isaac Sim

Install Isaac Sim separately, or reuse an existing installation:

```bash
export ISAAC=/path/to/isaacsim
mkdir -p runs
"$ISAAC/python.sh" build_scene.py --headless --save runs/scene.usd
```

The scene loads the FANUC ER-4iA robot USD from Isaac's asset root. Network access
or an available asset cache is needed on first use. Robot USD and its referenced
files are not bundled. An optional local asset/URI override is supported:

```bash
export FANUC_ROBOT_USD=/path/to/er4ia.usd
```

Preserve the asset's dependencies, original joint names and hierarchy.
To open the scene interactively, run `"$ISAAC/python.sh" build_scene.py`.

## 3. Cosmos3-Nano

Use a separate environment with CUDA-enabled PyTorch, compatible Transformers,
and NVIDIA's `transformers-cosmos3` package/framework. Obtain Cosmos3-Nano model
weights separately; these large dependencies are not in the starter.

```bash
export COSMOS=/path/to/cosmos-environment/bin/python
"$COSMOS" -m pip install -r requirements-cosmos.txt
export COSMOS3_NANO_PATH=/path/to/Cosmos3-Nano/snapshot
```

If the environment uses uv instead of pip:
`uv pip install --python "$COSMOS" -r requirements-cosmos.txt`.
The snapshot must contain `config.json` and the model/processor files.
If the framework package is not installed, set its checkout location:

```bash
export COSMOS3_ROOT=/path/to/cosmos-framework
```

That directory must contain `packages/transformers-cosmos3/transformers_cosmos3/`.
With no explicit snapshot, model discovery uses the Hugging Face cache;
`HF_HOME` or `HF_HUB_CACHE` can select its location.

Check dependencies and model loading before a complete run:

```bash
"$COSMOS" -c 'import torch, transformers, scipy, cv2; print("CUDA:", torch.cuda.is_available())'
"$COSMOS" -c 'from perceive import load_model; model, processor = load_model(); print("Model ready")'
```

CUDA must be available. The second command also checks framework imports and
weights, and exits after loading to release GPU memory.

## 4. Run and verify

```bash
./run_sort_pipeline.sh runs/stack_01
```

Use an empty run directory. Without an argument, the launcher creates a
timestamped directory. `HOST_PYTHON` optionally overrides `.venv/bin/python`.

| Stage | Interpreter | Entry point |
| --- | --- | --- |
| Calibrated 4K table capture | Isaac | `capture_scene.py` |
| Cube semantics and measured pose reporting | Cosmos | `perceive.py --stack-cubes` |
| Plan and evaluation | Host | `plan_sort.py` |
| Physical grasps and wrist control | Isaac | `sort_scene.py` |
| Locally generated review videos | Host | `encode_video.py` |

The run produces its own images, videos and JSON reports. Read
`execution_report.json` and require `task_complete: true`, six verified pickups
and six verified placements. Inspect `sort_plan.json` for skips and
`perception_evaluation.json` for position/colour/yaw errors. Raw model replies
and image-measured pose inputs are saved under `cube_pose_crops/`.
Square yaw errors use 90° symmetry. Ground truth evaluates results only.

To rerun physical validation without rendering, use a fresh captured/planned run:

```bash
"$ISAAC/python.sh" sort_scene.py --run runs/new_planned_run --physics-only
```

## Troubleshooting

- Missing launcher: check the interpreter paths and executable permissions.
- Missing model/framework: verify the snapshot and package/checkout paths.
- Missing robot: check Isaac's asset root or `FANUC_ROBOT_USD` and dependencies.
- No eligible picks: inspect skips and raw model replies; all six poses must pass.
- Incomplete stack: inspect the execution report and simulator logs.
- Changed geometry/layout/camera: recapture and replan in an empty directory.

Agents should report missing dependencies and completed checks precisely;
host tests do not prove model or simulator execution. Read `AGENTS.md` for
editing constraints and success criteria.
