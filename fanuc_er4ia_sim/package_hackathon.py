"""Build a participant release with an explicit file set and integrity manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
CODE = (
    "backproject.py",
    "build_scene.py",
    "capture_scene.py",
    "cell.py",
    "cube_pose.py",
    "encode_video.py",
    "fk.py",
    "gripper_geometry.py",
    "ik.py",
    "perceive.py",
    "plan_grasps.py",
    "plan_sort.py",
    "render_settings.py",
    "scene_config.py",
    "sort_scene.py",
    "test_cube_perception.py",
    "test_cube_pose.py",
    "test_sort_planning.py",
    "test_runtime_config.py",
    "test_support.py",
    "package_hackathon.py",
    "run_sort_pipeline.sh",
    "share.sh",
    "requirements.txt",
    "requirements-cosmos.txt",
    "README.md",
    "AGENTS.md",
    "bootstrap.sh",
    "make_textures.py",
    "SETUP.md",
    ".gitignore",
    "docs/MODEL_NOTES.md",
    "docs/CALIBRATION_V05.md",
)
ASSETS = (
    "assets/gripper/smc_mhzl2_16d2.json",
    "assets/gripper/photo_finger_detail.json",
    "assets/usd_assets/cube.usda",
    "assets/usd_assets/vial.usda",
    "assets/usd_assets/vial_cap.usda",
)


def package(output):
    """Bundle only starter source, instructions, tests and runtime geometry."""
    files = {Path(name): ROOT / name for name in CODE + ASSETS}
    missing = [str(source) for source in files.values() if not source.is_file()]
    if missing:
        raise FileNotFoundError("Missing release files:\n" + "\n".join(missing))
    output = output.resolve()
    if output in files.values() or any(
        output.is_relative_to(ROOT / d) for d in ("assets", "textures")
    ):
        raise ValueError("Choose an output outside the bundled asset directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "package": "FANUC ER-4iA hackathon starter",
        "media_included": False,
        "external_dependencies": [
            "Isaac Sim",
            "FANUC robot USD",
            "Cosmos3-Nano framework and weights",
        ],
        "files": [],
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative, source in sorted(files.items()):
            data = source.read_bytes()
            archive.write(source, "fanuc_er4ia_sim/" + relative.as_posix())
            manifest["files"].append(
                {
                    "path": relative.as_posix(),
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        archive.writestr(
            "fanuc_er4ia_sim/MANIFEST.json", json.dumps(manifest, indent=2) + "\n"
        )
    with zipfile.ZipFile(output) as archive:
        damaged = archive.testzip()
        if damaged:
            raise RuntimeError(f"ZIP integrity failure: {damaged}")
    print(
        f"Created {output} ({output.stat().st_size / 1024:.0f} KB; {len(files)} files + manifest)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT.parent / "fanuc_er4ia_hackathon_starter.zip"
    )
    args = parser.parse_args()
    package(args.output)


if __name__ == "__main__":
    main()
