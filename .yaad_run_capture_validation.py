import json
import re
import runpy
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/job/out")
PROJECT = Path("/job/in/project")
CAPTURE_SCRIPT = PROJECT / "capture_scene.py"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def progress(message):
    line = f"[progress] {timestamp()} {message}"
    print(line, flush=True)
    with (OUT / "capture_progress.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()


progress("capture harness started")
progress("before Isaac/Kit initialization")

import isaacsim

OriginalSimulationApp = isaacsim.SimulationApp


class ValidatingSimulationApp:
    def __init__(self, *args, **kwargs):
        self._app = OriginalSimulationApp(*args, **kwargs)
        progress("after Isaac/Kit initialization")

    def __getattr__(self, name):
        return getattr(self._app, name)

    def close(self, *args, **kwargs):
        progress("before capture-stage validation")
        report = {
            "validation": "isaac_capture_only",
            "status": "FAIL",
            "capture_script": str(CAPTURE_SCRIPT),
            "robot_usd": "/job/in/assets/isaac-6.0/Assets/Isaac/6.0/Isaac/Robots/Fanuc/er4ia/er4ia.usd",
            "automatic_asset_downloads_requested": False,
            "cosmos_run": False,
            "planning_run": False,
            "sorting_run": False,
            "full_pipeline_run": False,
            "checks": {},
            "failures": [],
        }
        validation_error = None

        try:
            import cv2
            import omni.usd
            from pxr import UsdGeom, UsdPhysics
            import cell
            import scene_config as C

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise RuntimeError("omni.usd returned no active capture stage")

            report["stage_prim_count"] = sum(1 for _ in stage.Traverse())

            robot = stage.GetPrimAtPath(C.ROBOT_PRIM)
            articulation_paths = [
                str(prim.GetPath())
                for prim in stage.Traverse()
                if str(prim.GetPath()).startswith(C.ROBOT_PRIM)
                and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            ]

            cube_paths = [
                f"/World/SortCube{i}" for i in range(len(C.SORT_CUBES))
            ]
            cap_paths = [
                f"/World/SortCap{i}" for i in range(len(C.SORT_CAPS))
            ]
            vial_paths = [
                f"/World/SortVial{i}" for i in range(len(C.SORT_VIALS))
            ]
            camera_paths = [
                cell.TABLE_CAM_PATH,
                cell.SIDE_CAM_PATH,
                cell.WRIST_CAM_PATH,
            ]
            light_paths = [
                "/World/DomeLight",
                "/World/KeyLight",
                "/World/FillLight",
                "/World/RimLight",
            ]

            def valid(path):
                return stage.GetPrimAtPath(path).IsValid()

            output_paths = {
                "image": OUT / "world_camera.png",
                "camera_metadata": OUT / "camera.json",
                "ground_truth": OUT / "ground_truth.json",
                "scene_state": OUT / "scene_state.json",
            }

            image = cv2.imread(str(output_paths["image"]))
            ground_truth = json.loads(output_paths["ground_truth"].read_text())
            scene_state = json.loads(output_paths["scene_state"].read_text())
            camera_metadata = json.loads(output_paths["camera_metadata"].read_text())

            report["robot"] = {
                "path": C.ROBOT_PRIM,
                "exists": robot.IsValid(),
                "articulation_paths": articulation_paths,
                "articulation_discovered": bool(articulation_paths),
            }
            report["cubes"] = {
                "expected": len(C.SORT_CUBES),
                "paths": cube_paths,
                "all_exist": all(valid(path) for path in cube_paths),
            }
            report["distractors"] = {
                "caps_expected": len(C.SORT_CAPS),
                "vials_expected": len(C.SORT_VIALS),
                "all_exist": all(valid(path) for path in cap_paths + vial_paths),
            }
            report["table"] = {
                "path": "/World/Table",
                "exists": valid("/World/Table"),
            }
            report["lighting"] = {
                "paths": light_paths,
                "all_exist": all(valid(path) for path in light_paths),
            }
            report["cameras"] = {
                "paths": camera_paths,
                "all_exist": all(
                    valid(path)
                    and bool(UsdGeom.Camera(stage.GetPrimAtPath(path)))
                    for path in camera_paths
                ),
                "captured_camera": camera_metadata.get("camera"),
            }
            report["capture"] = {
                "outputs": {
                    name: str(path) for name, path in output_paths.items()
                },
                "all_outputs_exist": all(
                    path.is_file() for path in output_paths.values()
                ),
                "image_width": int(image.shape[1]) if image is not None else 0,
                "image_height": int(image.shape[0]) if image is not None else 0,
                "ground_truth_objects": len(
                    ground_truth.get("objects", [])
                ),
                "scene_state_objects": len(
                    scene_state.get("objects", {})
                ),
            }

            checks = {
                "robot_exists": report["robot"]["exists"],
                "articulation_discovered": report["robot"][
                    "articulation_discovered"
                ],
                "six_cubes_exist": (
                    report["cubes"]["expected"] == 6
                    and report["cubes"]["all_exist"]
                ),
                "table_exists": report["table"]["exists"],
                "lighting_exists": report["lighting"]["all_exist"],
                "all_cameras_exist": report["cameras"]["all_exist"],
                "table_camera_captured": (
                    report["cameras"]["captured_camera"] == "table"
                ),
                "capture_outputs_exist": report["capture"][
                    "all_outputs_exist"
                ],
                "capture_image_nonempty": (
                    report["capture"]["image_width"] > 0
                    and report["capture"]["image_height"] > 0
                ),
                "ground_truth_present": (
                    report["capture"]["ground_truth_objects"] > 0
                ),
                "scene_state_present": (
                    report["capture"]["scene_state_objects"] > 0
                ),
            }
            report["checks"] = checks
            report["failures"] = [
                name for name, passed in checks.items() if not passed
            ]
            report["status"] = (
                "PASS" if not report["failures"] else "FAIL"
            )
        except Exception as exc:
            validation_error = exc
            report["failures"].append(
                f"{type(exc).__name__}: {exc}"
            )
            report["traceback"] = traceback.format_exc()
        finally:
            report["finished_utc"] = timestamp()
            (OUT / "capture_validation.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            progress(
                "after capture-stage validation; "
                f"status={report['status']} "
                f"failures={len(report['failures'])}"
            )
            progress("closing Isaac/Kit application")
            close_result = self._app.close(*args, **kwargs)
            progress("Isaac/Kit application closed")

        if validation_error is not None:
            raise validation_error
        if report["status"] != "PASS":
            raise RuntimeError(
                "capture validation failed: "
                + ", ".join(report["failures"])
            )
        return close_result


isaacsim.SimulationApp = ValidatingSimulationApp
sys.path.insert(0, str(PROJECT))
sys.argv = [
    str(CAPTURE_SCRIPT),
    "--out",
    str(OUT),
    "--camera",
    "table",
]

progress(
    "launching documented capture_scene.py workflow "
    "for one table-camera capture"
)
try:
    runpy.run_path(str(CAPTURE_SCRIPT), run_name="__main__")
except BaseException as exc:
    failure = {
        "status": "FAIL",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "finished_utc": timestamp(),
    }
    (OUT / "capture_harness_failure.json").write_text(
        json.dumps(failure, indent=2) + "\n",
        encoding="utf-8",
    )
    progress(f"capture workflow failed: {type(exc).__name__}: {exc}")
    raise

progress("capture workflow completed successfully")
