"""Execute real shell entrypoints against fake host/runtime commands, never a GPU.

These contracts prove argument, lifecycle and filesystem handling, not GNU
timeout's implementation, actual apt behavior, drivers, Kit, or physics.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from rc_car.workflow_config import load_workflow

ROOT = Path(__file__).resolve().parents[2]


def executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)


@pytest.fixture
def host(tmp_path: Path, workflow_payload: dict) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "copied repo"
    for directory in ("scripts", "outputs", "generated_assets", "src/rc_car"):
        (root / directory).mkdir(parents=True)
    for name in ("setup_runtime.sh", "run_gpu_checks.sh"):
        shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
    (root / "src/rc_car/provenance.py").write_text(
        (ROOT / "src/rc_car/provenance.py").read_text()
        + r"""

# The shell contract exercises the actual source verifier but substitutes the
# installed wheel checker; its real integrity checks have separate unit tests.
def inspect_isaaclab_runtime():
    import os
    if os.environ.get("FAKE_RUNTIME_FAILURE"):
        raise ValueError("Synthetic runtime identity failure")
    @dataclass(frozen=True)
    class FakeRuntime:
        distribution: str = "isaaclab"
        version: str = "2.3.0"
        package_root: str = "/synthetic/installed/runtime"
        files_verified: int = 968
        content_sha256: str = "synthetic-runtime-content-hash"
    return FakeRuntime()
"""
    )
    shutil.copy2(ROOT / "src/rc_car/workflow_config.py", root / "src/rc_car/workflow_config.py")
    (root / "config.yaml").write_text(yaml.safe_dump(workflow_payload))
    (root / "src/rc_car/__init__.py").write_text("")
    (root / "src/rc_car/experiments").mkdir()
    (root / "src/rc_car/experiments/__init__.py").write_text("")
    shutil.copy2(
        ROOT / "src/rc_car/experiments/config.py", root / "src/rc_car/experiments/config.py"
    )
    (root / "src/rc_car/experiments/online_check.py").write_text(r"""
import json
import os
import sys
with open(os.environ["FAKE_CALLS"], "a") as stream:
    stream.write(json.dumps({
        "name": "online_check", "args": sys.argv[1:],
        "mode": os.environ.get("WANDB_MODE"),
        "has_key": bool(os.environ.get("WANDB_API_KEY")),
        "resume_keys": [key for key in (
            "WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RESUME_FROM", "WANDB_FORK_FROM", "WANDB_NAME"
        ) if key in os.environ],
        "directories": {key: os.environ.get(key) for key in (
            "WANDB_DIR", "WANDB_DATA_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR",
            "WANDB_ARTIFACT_DIR"
        )}
    }) + "\n")
sys.exit(int(os.environ.get("FAKE_ONLINE_CHECK_EXIT", "0")))
""")
    runtime = root / ".venv/bin/python"
    # Keep the test interpreter's actual venv prefix so the copied YAML parser
    # sees its installed Pydantic/PyYAML dependencies. A relocated Python symlink
    # loses that prefix when no pyvenv.cfg is present beside it.
    executable(runtime, f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} "$@"\n')
    (root / "generated_assets/fixture.json").write_text('{"kind":"procedural-smoke-test-only"}')
    (root / "fixture-manifest.json").write_text("{}")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    command = (
        f"#!{sys.executable}\n"
        + r"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

name = Path(sys.argv[0]).name
with open(os.environ["FAKE_CALLS"], "a") as stream:
    stream.write(json.dumps({"name": name, "args": sys.argv[1:], "pid": os.getpid()}) + "\n")
if name == "uname":
    print(os.environ.get("FAKE_ARCH", "x86_64") if sys.argv[1] == "-m"
          else os.environ.get("FAKE_PLATFORM", "Linux"))
elif name == "flock":
    try:
        fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(1)
elif name == "timeout":
    # Emulate the process group for the supervisor/cleanup contract only.
    # The command layout is fixed and asserted separately below.
    if os.environ.get("FAKE_GROUP_DELAY"):
        time.sleep(3)
    os.setpgid(0, 0)
    if "FAKE_TIMEOUT_EXIT" in os.environ:
        sys.exit(int(os.environ["FAKE_TIMEOUT_EXIT"]))
    sys.exit(subprocess.call(sys.argv[4:], pass_fds=(9,)))
elif name == "git":
    # A transferred source capsule has provenance but is not a Git checkout.
    sys.exit(2)
elif name == "sudo":
    assert sys.argv[1] == "-n"
    sys.exit(subprocess.call(sys.argv[2:]))
elif name not in ("uv", "apt-get"):
    sys.exit(2)
"""
    )
    for name in ("uname", "timeout", "flock", "git", "uv", "apt-get", "sudo"):
        executable(fake_bin / name, command)
    shell_record = r"""#!/usr/bin/env bash
set -euo pipefail
printf '{"name":"%s"}\n' "$(basename "$0")" >> "$FAKE_CALLS"
exit "${FAKE_PREFLIGHT_EXIT:-0}"
"""
    executable(root / "scripts/gpu_preflight.sh", shell_record)
    executable(
        root / "scripts/setup_isaac.sh",
        shell_record.replace("FAKE_PREFLIGHT_EXIT", "FAKE_SETUP_EXIT"),
    )
    executable(
        root / "scripts/isaac.sh",
        '#!/usr/bin/env bash\nexec .venv/bin/python "$@"\n',
    )
    (root / "scripts/bundle.py").write_text(r"""
import json
import os
import sys
with open(os.environ["FAKE_CALLS"], "a") as stream:
    stream.write(json.dumps({"name": "bundle", "args": sys.argv[1:]}) + "\n")
sys.exit(int(os.environ.get("FAKE_BUNDLE_EXIT", "0")))
""")
    (root / "scripts/smoke_test.py").write_text(r"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
output = Path(sys.argv[sys.argv.index("--output") + 1])
output.mkdir()
if os.environ.get("FAKE_STATUS_WRITE_FAILURE"):
    (output.parent / "exit_status.json").mkdir()
record = {"name": "smoke", "args": sys.argv[1:],
          "env": {key: value for key, value in os.environ.items() if key.startswith("WANDB_")}}
with open(os.environ["FAKE_CALLS"], "a") as stream:
    stream.write(json.dumps(record) + "\n")
if os.environ.get("FAKE_DESCENDANT"):
    subprocess.Popen([sys.executable, "-c", "import os,time; from pathlib import Path; time.sleep(0.6); Path(os.environ['FAKE_DESCENDANT']).touch()"])
if os.environ.get("FAKE_HOLD"):
    time.sleep(30)
print("fake GPU payload ran", flush=True)
sys.exit(int(os.environ.get("FAKE_PAYLOAD_EXIT", "0")))
""")
    executable(
        root / "scripts/run_experiment.sh",
        r"""#!/usr/bin/env bash
set -euo pipefail
mkdir "$1"
printf '{"name":"rehearsal","recipe":"%s","tracking_mode":"%s"}\n' "$2" "$WANDB_MODE" >> "$FAKE_CALLS"
exit "${FAKE_PAYLOAD_EXIT:-0}"
""",
    )
    covered = (
        "scripts/run_gpu_checks.sh",
        "scripts/setup_runtime.sh",
        "scripts/smoke_test.py",
        "config.yaml",
    )
    (root / "outputs/source-provenance.json").write_text(
        json.dumps(
            {
                "files_sha256": {
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in covered
                }
            }
        )
    )
    env = dict(os.environ)
    env.update(
        {
            "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
            "PYTHONPATH": str(root / "src"),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "FAKE_CALLS": str(tmp_path / "calls.jsonl"),
        }
    )
    return root, env


def calls(env: dict[str, str]) -> list[dict[str, object]]:
    path = Path(env["FAKE_CALLS"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def command(root: Path, mode: str = "smoke-basic", output: str = "outputs/job") -> list[str]:
    return [
        "bash",
        str(root / "scripts/run_gpu_checks.sh"),
        "--mode",
        mode,
        "--output",
        output,
        "--fixture-manifest",
        "fixture-manifest.json",
    ]


def run(arguments: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, env=env, text=True, capture_output=True, timeout=10)


@pytest.mark.parametrize("script", ["setup_runtime.sh", "run_gpu_checks.sh"])
def test_plan_is_mac_safe_and_does_not_touch_host(
    host: tuple[Path, dict[str, str]], script: str
) -> None:
    root, env = host
    env.pop("OMNI_KIT_ACCEPT_EULA")
    env["FAKE_PLATFORM"] = "Darwin"
    args = (
        command(root) if script == "run_gpu_checks.sh" else ["bash", str(root / "scripts" / script)]
    )
    result = run([*args, "--plan"], env)
    assert result.returncode == 0, result.stderr
    assert "PLAN ONLY" in result.stdout
    assert calls(env) == []
    assert not (root / ".gpu-job.lock").exists()
    assert not (root / "outputs/job").exists()


@pytest.mark.parametrize(
    "mode,task,count",
    [
        ("smoke-basic", "Basic-Car-Training-v0", "12"),
        ("smoke-multi", "Multiagent-Car-Training-v0", "4"),
    ],
)
def test_smoke_contract_and_offline_isolation(
    host: tuple[Path, dict[str, str]], mode: str, task: str, count: str
) -> None:
    root, env = host
    env.update(
        {"WANDB_MODE": "online", "WANDB_API_KEY": "secret-must-not-leak", "WANDB_RUN_ID": "old"}
    )
    result = run(command(root, mode), env)
    assert result.returncode == 0, result.stderr + (root / "outputs/job/job.log").read_text()
    records = calls(env)
    smoke = next(record for record in records if record["name"] == "smoke")
    args = smoke["args"]
    assert isinstance(args, list)
    assert args[args.index("--task") + 1] == task
    assert args[args.index("--num_envs") + 1] == count
    assert "--train" in args and "--fixture" in args and "--enable_cameras" in args
    wandb = smoke["env"]
    assert isinstance(wandb, dict)
    assert wandb["WANDB_MODE"] == "offline"
    assert "WANDB_API_KEY" not in wandb and "WANDB_RUN_ID" not in wandb
    for name in (
        "WANDB_DIR",
        "WANDB_DATA_DIR",
        "WANDB_CACHE_DIR",
        "WANDB_CONFIG_DIR",
        "WANDB_ARTIFACT_DIR",
    ):
        assert Path(wandb[name]).is_relative_to(root / "outputs/job")
    bundle = next(record for record in records if record["name"] == "bundle")
    assert isinstance(bundle["args"], list)
    assert "--require-fixtures" in bundle["args"]
    assert [record["name"] for record in records].index("bundle") < [
        record["name"] for record in records
    ].index("gpu_preflight.sh")
    deadline = next(record for record in records if record["name"] == "timeout")
    assert isinstance(deadline["args"], list)
    assert deadline["args"][:3] == ["--signal=TERM", "--kill-after=30", "900"]
    output = root / "outputs/job"
    assert json.loads((output / "exit_status.json").read_text())["exit_code"] == 0
    identity = json.loads((output / "runtime-identity.json").read_text())
    assert identity["version"] == "2.3.0"
    assert identity["content_sha256"] == "synthetic-runtime-content-hash"
    assert (output / "source-provenance.json").is_file()
    assert (output / "fixture-bundle-manifest.json").is_file()
    assert "secret-must-not-leak" not in (output / "job.log").read_text()


@pytest.mark.parametrize(
    "mode,task",
    [
        ("rehearsal-basic", "Basic-Car-Training-v0"),
        ("rehearsal-multi", "Multiagent-Car-Training-v0"),
    ],
)
def test_rehearsal_contract(host: tuple[Path, dict[str, str]], mode: str, task: str) -> None:
    root, env = host
    result = run(command(root, mode), env)
    assert result.returncode == 0, result.stderr
    entry = next(record for record in calls(env) if record["name"] == "rehearsal")
    recipe_path = root / "outputs/job/requested-recipe.json"
    assert entry["recipe"] == str(recipe_path)
    recipe = json.loads(recipe_path.read_text())
    assert recipe["task"] == task
    assert recipe["wandb"]["mode"] == "offline"
    expected = load_workflow(root / "config.yaml").requested_recipe(
        "basic-smoke" if mode == "rehearsal-basic" else "multiagent-smoke"
    )
    expected["wandb"]["mode"] = "offline"
    assert recipe == expected
    assert entry["tracking_mode"] == "offline"
    assert not any(record["name"] == "online_check" for record in calls(env))
    assert (root / "outputs/job/experiment").is_dir()


@pytest.mark.parametrize("recaptured", [False, True])
def test_rehearsal_rejects_changed_or_invalid_yaml_before_gpu(
    host: tuple[Path, dict[str, str]], recaptured: bool
) -> None:
    root, env = host
    workflow = root / "config.yaml"
    workflow.write_text("invalid: workflow\n")
    if recaptured:
        provenance = root / "outputs/source-provenance.json"
        source = json.loads(provenance.read_text())
        source["files_sha256"]["config.yaml"] = hashlib.sha256(workflow.read_bytes()).hexdigest()
        provenance.write_text(json.dumps(source))
    result = run(command(root, "rehearsal-basic"), env)
    assert result.returncode != 0
    assert not (root / "outputs/job/requested-recipe.json").exists()
    assert not any(
        record["name"] in {"gpu_preflight.sh", "online_check", "rehearsal"} for record in calls(env)
    )


def test_generated_experiment_recipe_uses_the_same_bounded_supervisor(
    host: tuple[Path, dict[str, str]],
) -> None:
    root, env = host
    recipe = root / "outputs/requested-recipe.json"
    recipe.write_text("{}")
    result = run([*command(root, "experiment"), "--config", "outputs/requested-recipe.json"], env)
    assert result.returncode == 0, result.stderr
    records = calls(env)
    entry = next(record for record in records if record["name"] == "rehearsal")
    assert entry["recipe"] == str(recipe)
    assert entry["tracking_mode"] == "offline"
    assert any(record["name"] == "timeout" for record in records)
    status = json.loads((root / "outputs/job/exit_status.json").read_text())
    assert status["mode"] == "experiment" and status["exit_code"] == 0


@pytest.mark.parametrize("bad", ["absent", "symlink", "missing-argument", "fixed-mode"])
def test_generated_recipe_guard_fails_before_gpu(
    host: tuple[Path, dict[str, str]], bad: str
) -> None:
    root, env = host
    args = command(root, "smoke-basic" if bad == "fixed-mode" else "experiment")
    if bad != "missing-argument":
        args.extend(["--config", "outputs/recipe.json"])
    if bad == "symlink":
        (root / "outputs/recipe.json").symlink_to(root / "fixture-manifest.json")
    result = run(args, env)
    assert result.returncode == 2
    assert not any(
        record["name"] in {"timeout", "smoke", "rehearsal", "gpu_preflight.sh"}
        for record in calls(env)
    )


@pytest.mark.parametrize("mode", ["rehearsal-basic", "rehearsal-multi"])
def test_online_preflight_before_gpu_and_isolated_fresh_run(
    host: tuple[Path, dict[str, str]], mode: str
) -> None:
    root, env = host
    secret = "test-only-key-must-not-be-logged"
    env.update(
        {
            "WANDB_MODE": "offline",
            "WANDB_API_KEY": secret,
            "WANDB_RUN_ID": "old",
            "WANDB_RESUME": "must",
            "WANDB_RESUME_FROM": "old?step=1",
            "WANDB_FORK_FROM": "old?step=2",
            "WANDB_NAME": "old-display-name",
        }
    )
    result = run([*command(root, mode), "--tracking-mode", "online"], env)
    assert result.returncode == 0, result.stderr
    records = calls(env)
    names = [record["name"] for record in records]
    assert names.index("bundle") < names.index("online_check") < names.index("gpu_preflight.sh")
    assert names.index("gpu_preflight.sh") < names.index("rehearsal")
    check = next(record for record in records if record["name"] == "online_check")
    recipe_path = root / "outputs/job/requested-recipe.json"
    assert check["args"] == ["--config", str(recipe_path)]
    assert json.loads(recipe_path.read_text())["wandb"]["mode"] == "online"
    assert check["mode"] == "online" and check["has_key"] is True
    assert check["resume_keys"] == []
    directories = check["directories"]
    assert isinstance(directories, dict)
    assert all(Path(value).is_relative_to(root / "outputs/job") for value in directories.values())
    rehearsal = next(record for record in records if record["name"] == "rehearsal")
    assert rehearsal["tracking_mode"] == "online"
    output = root / "outputs/job"
    assert json.loads((output / "exit_status.json").read_text())["tracking_mode"] == "online"
    assert secret not in result.stdout + result.stderr + (output / "job.log").read_text()
    assert secret not in Path(env["FAKE_CALLS"]).read_text()


def test_online_auth_failure_stops_before_gpu_without_offline_fallback(
    host: tuple[Path, dict[str, str]],
) -> None:
    root, env = host
    env["FAKE_ONLINE_CHECK_EXIT"] = "17"
    result = run([*command(root, "rehearsal-basic"), "--tracking-mode", "online"], env)
    assert result.returncode == 17
    names = [record["name"] for record in calls(env)]
    assert "online_check" in names
    assert not any(name in names for name in ("gpu_preflight.sh", "rehearsal", "smoke"))
    output = root / "outputs/job"
    status = json.loads((output / "exit_status.json").read_text())
    assert status["exit_code"] == 17 and status["tracking_mode"] == "online"
    assert "--tracking-mode offline" in (output / "job.log").read_text()
    assert "fresh output path" in (output / "job.log").read_text()


def test_online_plan_does_not_authenticate_or_create_output(
    host: tuple[Path, dict[str, str]],
) -> None:
    root, env = host
    result = run([*command(root, "rehearsal-basic"), "--tracking-mode", "online", "--plan"], env)
    assert result.returncode == 0, result.stderr
    assert "WANDB_MODE=online" in result.stdout
    assert "before GPU preflight or Kit" in result.stdout
    assert calls(env) == []
    assert not (root / "outputs/job").exists()


@pytest.mark.parametrize(
    "mode,tracking_mode",
    [("smoke-basic", "online"), ("smoke-multi", "online"), ("rehearsal-basic", "auto")],
)
def test_rejects_unsupported_tracking_mode_before_host_access(
    host: tuple[Path, dict[str, str]], mode: str, tracking_mode: str
) -> None:
    root, env = host
    result = run([*command(root, mode), "--tracking-mode", tracking_mode], env)
    assert result.returncode == 2
    assert calls(env) == []
    assert not (root / "outputs/job").exists()


@pytest.mark.parametrize(
    "bad", ["platform", "architecture", "eula", "asset-root", "timeout", "existing", "symlink"]
)
def test_refuses_unsafe_launch_before_payload(host: tuple[Path, dict[str, str]], bad: str) -> None:
    root, env = host
    args = command(root)
    if bad == "platform":
        env["FAKE_PLATFORM"] = "Darwin"
    elif bad == "architecture":
        env["FAKE_ARCH"] = "aarch64"
    elif bad == "eula":
        env.pop("OMNI_KIT_ACCEPT_EULA")
    elif bad == "asset-root":
        args += ["--asset-root", "elsewhere"]
    elif bad == "timeout":
        args += ["--timeout-seconds", "0"]
    elif bad == "existing":
        (root / "outputs/job").mkdir()
        (root / "outputs/job/user-file").write_text("preserve")
    else:
        (root / "outputs/job").symlink_to(root / "generated_assets", target_is_directory=True)
    result = run(args, env)
    assert result.returncode == 2
    assert not any(
        record["name"] in ("smoke", "rehearsal", "gpu_preflight.sh", "timeout")
        for record in calls(env)
    )
    if bad == "existing":
        assert (root / "outputs/job/user-file").read_text() == "preserve"


@pytest.mark.parametrize("failure", ["source", "runtime", "fixture"])
def test_identity_failure_is_logged_before_gpu(
    host: tuple[Path, dict[str, str]], failure: str
) -> None:
    root, env = host
    if failure == "source":
        with (root / "scripts/smoke_test.py").open("a") as stream:
            stream.write("\n# changed after snapshot\n")
    elif failure == "runtime":
        env["FAKE_RUNTIME_FAILURE"] = "yes"
    else:
        env["FAKE_BUNDLE_EXIT"] = "19"
    result = run(command(root), env)
    assert result.returncode != 0
    assert (
        json.loads((root / "outputs/job/exit_status.json").read_text())["exit_code"]
        == result.returncode
    )
    assert not any(record["name"] in ("smoke", "gpu_preflight.sh") for record in calls(env))


@pytest.mark.parametrize(
    "control,code",
    [("FAKE_PAYLOAD_EXIT", 7), ("FAKE_PREFLIGHT_EXIT", 9), ("FAKE_TIMEOUT_EXIT", 124)],
)
def test_failure_and_timeout_status_preserved(
    host: tuple[Path, dict[str, str]], control: str, code: int
) -> None:
    root, env = host
    env[control] = str(code)
    result = run(command(root), env)
    assert result.returncode == code
    assert json.loads((root / "outputs/job/exit_status.json").read_text())["exit_code"] == code


def await_smoke(env: dict[str, str]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if any(record["name"] == "smoke" for record in calls(env)):
            return
        time.sleep(0.02)
    raise AssertionError("Fake smoke did not start")


@pytest.mark.parametrize("payload_code,final_code", [(7, 7), (0, 74)])
def test_status_write_failure_preserves_primary_error(
    host: tuple[Path, dict[str, str]], payload_code: int, final_code: int
) -> None:
    root, env = host
    env["FAKE_PAYLOAD_EXIT"] = str(payload_code)
    env["FAKE_STATUS_WRITE_FAILURE"] = "yes"
    result = run(command(root), env)
    assert result.returncode == final_code
    assert "Failed to persist exit_status.json" in result.stderr


def test_real_rehearsal_runner_stays_in_outer_supervised_group(
    host: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    root, env = host
    shutil.copy2(ROOT / "scripts/run_experiment.sh", root / "scripts/run_experiment.sh")
    executable(
        root / "scripts/isaac.sh",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == -m ]]; then exit 0; fi
exec .venv/bin/python "$@"
""",
    )
    (root / "scripts/training_experiment.py").write_text("""
import json
import os
import subprocess
import sys
import time
with open(os.environ["FAKE_CALLS"], "a") as stream:
    stream.write(json.dumps({"name": "phase-started"}) + "\\n")
subprocess.Popen([sys.executable, "-c", "import os,time; from pathlib import Path; time.sleep(0.6); Path(os.environ['FAKE_DESCENDANT']).touch()"])
time.sleep(30)
""")
    marker = tmp_path / "rehearsal-descendant-survived"
    env["FAKE_DESCENDANT"] = str(marker)
    process = subprocess.Popen(
        command(root, "rehearsal-basic"),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not any(record["name"] == "phase-started" for record in calls(env)):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        # A nested GNU timeout normally creates a separate group. There must be
        # exactly one timeout: the outer ops supervisor, never an inner phase one.
        assert sum(record["name"] == "timeout" for record in calls(env)) == 1
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=3)
        assert process.returncode == 143
        time.sleep(0.8)
        assert not marker.exists()
    finally:
        # Keep the test safe even if a future regression reintroduces inner groups.
        for record in calls(env):
            if record["name"] == "timeout" and isinstance(record["pid"], int):
                try:
                    os.killpg(record["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=5)


def test_one_job_lock_and_signal_exit_status(host: tuple[Path, dict[str, str]]) -> None:
    root, env = host
    env["FAKE_HOLD"] = "yes"
    process = subprocess.Popen(
        command(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        await_smoke(env)
        second = run(command(root, output="outputs/second-job"), env)
        assert second.returncode == 75
        assert not (root / "outputs/second-job").exists()
        setup = run(["bash", str(root / "scripts/setup_runtime.sh")], env)
        assert setup.returncode == 75
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=5)
        assert process.returncode == 143
        assert json.loads((root / "outputs/job/exit_status.json").read_text())["exit_code"] == 143
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=5)


def test_completion_kills_lingering_process_group_member(
    host: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    root, env = host
    marker = tmp_path / "escaped-work-ran"
    env["FAKE_DESCENDANT"] = str(marker)
    result = run(command(root), env)
    assert result.returncode == 0
    time.sleep(0.8)
    assert not marker.exists()


def test_signal_before_timeout_creates_group_does_not_hang(
    host: tuple[Path, dict[str, str]],
) -> None:
    root, env = host
    env["FAKE_GROUP_DELAY"] = "yes"
    process = subprocess.Popen(
        command(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 3
        while not any(record["name"] == "timeout" for record in calls(env)):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=2)
        assert process.returncode == 143
        assert not any(record["name"] == "smoke" for record in calls(env))
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=5)


@pytest.mark.parametrize("install_deps", [False, True])
def test_setup_reuses_existing_scripts_and_optional_apt(
    host: tuple[Path, dict[str, str]], install_deps: bool
) -> None:
    root, env = host
    args = ["bash", str(root / "scripts/setup_runtime.sh")]
    if install_deps:
        args.append("--install-system-deps")
    result = run(args, env)
    assert result.returncode == 0, result.stderr
    records = calls(env)
    names = [record["name"] for record in records]
    assert names.index("gpu_preflight.sh") < names.index("setup_isaac.sh")
    apt = [record for record in records if record["name"] == "apt-get"]
    assert bool(apt) is install_deps
    if install_deps:
        assert apt[0]["args"] == ["update"]
        assert isinstance(apt[1]["args"], list)
        assert "vulkan-tools" in apt[1]["args"]
        assert "ffmpeg" in apt[1]["args"]
        assert not any("nvidia" in argument or "driver" in argument for argument in apt[1]["args"])


@pytest.mark.parametrize("script", ["setup_runtime.sh", "run_gpu_checks.sh"])
def test_shell_syntax(script: str) -> None:
    result = subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
