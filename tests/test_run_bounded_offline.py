from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_bounded_offline as bounded


class ResolveRootTests(unittest.TestCase):
    def test_requires_an_absolute_existing_directory(self) -> None:
        with self.assertRaisesRegex(bounded.ConfigurationError, "absolute"):
            bounded.resolve_root("relative/path")

    def test_rejects_an_ancestor_of_the_wrapper_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp).resolve()
            repo = parent / "exact-repo"
            repo.mkdir()

            with self.assertRaisesRegex(bounded.ConfigurationError, "ancestor"):
                bounded.resolve_root(str(parent), repo_root=repo)

    def test_resolves_a_symlink_before_mount_construction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            target = base / "target"
            link = base / "link"
            repo.mkdir()
            target.mkdir()
            link.symlink_to(target, target_is_directory=True)

            self.assertEqual(
                bounded.resolve_root(str(link), repo_root=repo), target.resolve()
            )

    def test_rejects_mount_syntax_delimiters(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bounded,root-") as tmp:
            with self.assertRaisesRegex(bounded.ConfigurationError, "comma"):
                bounded.resolve_root(tmp, repo_root=Path(tmp) / "unrelated-repo")

    def test_rejects_a_symlink_whose_resolved_target_contains_a_mount_delimiter(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as link_tmp,
            tempfile.TemporaryDirectory(prefix="bounded,target-") as target_tmp,
        ):
            link_base = Path(link_tmp)
            repo = link_base / "repo"
            link = link_base / "safe-link"
            target = Path(target_tmp) / "target"
            repo.mkdir()
            target.mkdir()
            link.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(
                bounded.ConfigurationError, "resolved --root.*comma"
            ):
                bounded.resolve_root(str(link), repo_root=repo)


class ArgumentAndExclusionTests(unittest.TestCase):
    def test_timeout_and_hard_limits_are_bounded(self) -> None:
        parser = bounded.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--root", "/tmp/example", "--timeout-seconds", "0", "tests"]
            )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--root", "/tmp/example", "--read-mib-per-sec", "64", "tests"]
            )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--root", "/tmp/example", "--read-iops", "1000", "tests"]
            )

    def test_defaults_exclude_vcs_worktrees_caches_envs_generated_and_large_data(
        self,
    ) -> None:
        args = bounded.build_parser().parse_args(["--root", "/tmp/example", "tests"])
        globs = bounded.exclusion_globs(args)

        for expected in (
            "!**/.git/**",
            "!**/worktrees/**",
            "!**/archive/**",
            "!**/.pytest_cache/**",
            "!**/.venv/**",
            "!**/artifacts/**",
            "!**/datasets/**",
            "!**/processed/**",
            "!*.db*",
            "!*.csv",
        ):
            self.assertIn(expected, globs)

    def test_explicit_category_inclusion_removes_only_that_exclusion_group(
        self,
    ) -> None:
        args = bounded.build_parser().parse_args(
            [
                "--root",
                "/tmp/example",
                "--include-archives",
                "--include-generated",
                "--include-large-data",
                "tests",
            ]
        )
        globs = bounded.exclusion_globs(args)

        self.assertNotIn("!**/archive/**", globs)
        self.assertNotIn("!**/artifacts/**", globs)
        self.assertNotIn("!**/datasets/**", globs)
        self.assertNotIn("!*.db*", globs)
        self.assertIn("!**/.git/**", globs)
        self.assertIn("!**/.venv/**", globs)

    def test_rg_pattern_must_not_be_empty(self) -> None:
        parser = bounded.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--root", "/tmp/example", "rg", ""])


class HostSupportTests(unittest.TestCase):
    def test_image_id_must_be_an_immutable_sha256_digest(self) -> None:
        image_id = "sha256:" + "a" * 64

        self.assertEqual(bounded._validated_image_id(image_id), image_id)
        for invalid in ("alpine:3.20", "sha256:abc", "sha256:" + "G" * 64):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    bounded.ConfigurationError, "immutable image ID"
                ):
                    bounded._validated_image_id(invalid)

    def test_host_probe_has_a_timeout(self) -> None:
        result = subprocess.CompletedProcess(["docker", "info"], 0, "ok", "")

        with mock.patch.object(
            bounded.subprocess, "run", return_value=result
        ) as runner:
            self.assertEqual(bounded._capture(["docker", "info"]), result)

        self.assertEqual(runner.call_args.kwargs["timeout"], 15)

    def test_host_probe_timeout_fails_as_configuration_error(self) -> None:
        with mock.patch.object(
            bounded.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["docker", "info"], 15),
        ):
            with self.assertRaisesRegex(bounded.ConfigurationError, "timed out"):
                bounded._capture(["docker", "info"])

    def test_host_probe_os_error_fails_as_configuration_error(self) -> None:
        with mock.patch.object(
            bounded.subprocess,
            "run",
            side_effect=OSError("docker executable disappeared"),
        ):
            with self.assertRaisesRegex(bounded.ConfigurationError, "could not run"):
                bounded._capture(["docker", "info"])


class BootstrapGuardTests(unittest.TestCase):
    def run_bootstrap(self, limit_line: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            cgroup_root = Path(tmp) / "cgroup"
            fake_bin = Path(tmp) / "bin"
            cgroup_root.mkdir()
            fake_bin.mkdir()
            (cgroup_root / "io.max").write_text(limit_line + "\n", encoding="utf-8")
            (cgroup_root / "io.weight").write_text("default 1\n", encoding="utf-8")
            (cgroup_root / "cpu.weight").write_text("8\n", encoding="utf-8")
            ionice = fake_bin / "ionice"
            ionice.write_text(
                '#!/bin/sh\nif [ "$1" = "-p" ]; then echo idle; fi\nexit 0\n',
                encoding="utf-8",
            )
            ionice.chmod(0o755)
            renice = fake_bin / "renice"
            renice.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            renice.chmod(0o755)
            env = dict(os.environ)
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            return subprocess.run(
                [
                    "/bin/sh",
                    "-eu",
                    "-c",
                    bounded.HARD_LIMIT_BOOTSTRAP,
                    "bounded-bootstrap",
                    "259:2",
                    "8388608",
                    "64",
                    str(cgroup_root),
                    "/bin/true",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

    def test_accepts_exact_reordered_limit_tokens(self) -> None:
        result = self.run_bootstrap("259:2 riops=64 wiops=max rbps=8388608 wbps=max")

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_non_exact_limit_tokens(self) -> None:
        for limit_line in (
            "259:2 rbps=83886080 wbps=max riops=640 wiops=max",
            "259:2 rbps=8388608x wbps=max riops=64x wiops=max",
            "259:2 xrbps=8388608 wbps=max riops=64 wiops=max",
            "259:2 rbps=8388608 wbps=max xriops=64 wiops=max",
            "259:2 wbps=max wiops=max",
            "259:2 rbps=8388608 rbps=8388608 riops=64 wiops=max",
            "259:2 rbps=8388608 wbps=max riops=64 riops=64 wiops=max",
        ):
            with self.subTest(limit_line=limit_line):
                result = self.run_bootstrap(limit_line)
                self.assertEqual(result.returncode, 78, result.stderr)


class CommandConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.host = bounded.HostSupport(
            docker=Path("/usr/bin/docker"),
            image="sha256:" + "a" * 64,
            rg_binary=Path("/opt/host/rg"),
            device=Path("/dev/nvme0n1"),
            device_id="259:2",
        )

    def test_rg_is_fixed_string_single_thread_and_container_root_only(self) -> None:
        args = bounded.build_parser().parse_args(
            ["--root", "/tmp/exact", "rg", "needle", "--glob", "*.md"]
        )
        command = bounded.workload_command(args)

        self.assertEqual(command[0], bounded.CONTAINER_RG)
        self.assertIn("--threads", command)
        self.assertEqual(command[command.index("--threads") + 1], "1")
        self.assertIn("--fixed-strings", command)
        self.assertEqual(command[-1], bounded.CONTAINER_ROOT)
        self.assertNotIn("/tmp/exact", command)
        self.assertLess(command.index("*.md"), command.index("!**/.git/**"))

    def test_test_discovery_is_a_file_only_single_thread_operation(self) -> None:
        args = bounded.build_parser().parse_args(["--root", "/tmp/exact", "tests"])
        command = bounded.workload_command(args)

        self.assertIn("--files", command)
        self.assertIn("test_*.py", command)
        self.assertIn("*_test.py", command)
        self.assertEqual(command[command.index("--threads") + 1], "1")

    def test_hashing_materializes_then_hashes_one_process_sequentially(self) -> None:
        args = bounded.build_parser().parse_args(
            ["--root", "/tmp/exact", "hash", "--glob", "*.md"]
        )
        command = bounded.workload_command(args)

        self.assertEqual(command[:3], ["/bin/sh", "-eu", "-c"])
        self.assertIn("manifest=/tmp/bounded-files", command[3])
        self.assertIn("xargs -0 -r sha256sum", command[3])
        self.assertIn("--threads", command)
        self.assertEqual(command[command.index("--threads") + 1], "1")

    def test_docker_command_is_read_only_low_priority_and_hard_limited(self) -> None:
        args = bounded.build_parser().parse_args(["--root", "/tmp/exact", "tests"])
        command = bounded.docker_command(
            args,
            root=Path("/tmp/exact"),
            host=self.host,
            workload=[bounded.CONTAINER_RG, "--version"],
            container_name="bounded-test",
        )

        self.assertIn("--read-only", command)
        self.assertIn("--network", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("--cpu-shares", command)
        self.assertEqual(command[command.index("--cpu-shares") + 1], "32")
        self.assertIn("--blkio-weight", command)
        self.assertEqual(command[command.index("--blkio-weight") + 1], "10")
        self.assertIn("/dev/nvme0n1:8388608", command)
        self.assertIn("/dev/nvme0n1:64", command)
        self.assertIn(
            "type=bind,src=/tmp/exact,dst=/scan,readonly,bind-recursive=disabled",
            command,
        )
        self.assertIn(
            "type=bind,src=/opt/host/rg,dst=/opt/bounded/rg,readonly", command
        )
        self.assertIn(self.host.image, command)
        self.assertNotIn(bounded.DEFAULT_IMAGE, command)
        timeout_index = command.index("timeout")
        self.assertLess(timeout_index, command.index("/bin/sh"))
        self.assertEqual(command[timeout_index + 5], str(args.timeout_seconds))
        bootstrap = command[command.index("/bin/sh") + 3]
        self.assertIn("io_max=$cgroup_root/io.max", bootstrap)
        self.assertIn(bounded.CONTAINER_CGROUP_ROOT, command)
        self.assertIn("exit 78", bootstrap)
        self.assertIn("ionice -c 3", bootstrap)
        self.assertIn("ionice_state=$(ionice -p $$)", bootstrap)
        self.assertIn("renice 15", bootstrap)

    def test_docker_command_rejects_a_mount_source_delimiter(self) -> None:
        args = bounded.build_parser().parse_args(["--root", "/tmp/exact", "tests"])
        host = bounded.HostSupport(
            docker=self.host.docker,
            image=self.host.image,
            rg_binary=Path("/opt/host/rg,unsafe"),
            device=self.host.device,
            device_id=self.host.device_id,
        )

        with self.assertRaisesRegex(
            bounded.ConfigurationError, "resolved rg path.*comma"
        ):
            bounded.docker_command(
                args,
                root=Path("/tmp/exact"),
                host=host,
                workload=[bounded.CONTAINER_RG, "--version"],
                container_name="bounded-test",
            )


class ExecutionTests(unittest.TestCase):
    class FakeProcess:
        def __init__(
            self, first_error: BaseException | None = None, returncode: int = 0
        ) -> None:
            self.first_error = first_error
            self.returncode = returncode
            self.wait_calls = 0
            self.terminated = False
            self.killed = False

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1 and self.first_error is not None:
                raise self.first_error
            return self.returncode

        def poll(self) -> int | None:
            return None if self.wait_calls == 1 else self.returncode

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    def test_timeout_cleans_container_and_returns_124(self) -> None:
        process = self.FakeProcess(
            subprocess.TimeoutExpired(["docker"], 1), returncode=-15
        )
        cleanup = mock.Mock()

        result = bounded.execute_docker(
            ["docker", "run"],
            container_name="bounded-timeout",
            timeout_seconds=1,
            popen_factory=mock.Mock(return_value=process),
            cleanup=cleanup,
        )

        self.assertEqual(result, 124)
        cleanup.assert_called_once_with("bounded-timeout")

    def test_interrupt_cleans_container_and_returns_130(self) -> None:
        process = self.FakeProcess(KeyboardInterrupt(), returncode=-15)
        cleanup = mock.Mock()

        result = bounded.execute_docker(
            ["docker", "run"],
            container_name="bounded-interrupt",
            timeout_seconds=30,
            popen_factory=mock.Mock(return_value=process),
            cleanup=cleanup,
        )

        self.assertEqual(result, 130)
        cleanup.assert_called_once_with("bounded-interrupt")

    def test_hard_limit_guard_failure_is_returned_without_fallback(self) -> None:
        process = self.FakeProcess(returncode=78)
        cleanup = mock.Mock()

        result = bounded.execute_docker(
            ["docker", "run"],
            container_name="bounded-hard-limit",
            timeout_seconds=30,
            popen_factory=mock.Mock(return_value=process),
            cleanup=cleanup,
        )

        self.assertEqual(result, 78)
        cleanup.assert_not_called()

    def test_cleanup_container_absorbs_docker_control_failures(self) -> None:
        with (
            mock.patch.object(bounded.shutil, "which", return_value="/usr/bin/docker"),
            mock.patch.object(
                bounded.subprocess,
                "run",
                side_effect=(
                    subprocess.TimeoutExpired(["docker", "stop"], 10),
                    OSError("docker rm unavailable"),
                ),
            ),
        ):
            self.assertFalse(bounded.cleanup_container("bounded-cleanup-failure"))

    def test_cleanup_container_attempts_remove_after_stop_failure(self) -> None:
        removed = subprocess.CompletedProcess(["docker", "rm"], 0)
        with (
            mock.patch.object(bounded.shutil, "which", return_value="/usr/bin/docker"),
            mock.patch.object(
                bounded.subprocess,
                "run",
                side_effect=(
                    subprocess.TimeoutExpired(["docker", "stop"], 10),
                    removed,
                ),
            ) as runner,
        ):
            self.assertTrue(bounded.cleanup_container("bounded-cleanup-recovery"))

        self.assertEqual(runner.call_count, 2)

    def test_timeout_settles_client_when_cleanup_raises(self) -> None:
        process = self.FakeProcess(
            subprocess.TimeoutExpired(["docker"], 1), returncode=-15
        )

        result = bounded.execute_docker(
            ["docker", "run"],
            container_name="bounded-cleanup-error",
            timeout_seconds=1,
            popen_factory=mock.Mock(return_value=process),
            cleanup=mock.Mock(
                side_effect=subprocess.TimeoutExpired(["docker", "stop"], 10)
            ),
        )

        self.assertEqual(result, 124)
        self.assertEqual(process.wait_calls, 2)

    def test_interrupt_settles_client_when_cleanup_raises(self) -> None:
        process = self.FakeProcess(KeyboardInterrupt(), returncode=-15)

        result = bounded.execute_docker(
            ["docker", "run"],
            container_name="bounded-interrupt-cleanup-error",
            timeout_seconds=30,
            popen_factory=mock.Mock(return_value=process),
            cleanup=mock.Mock(side_effect=OSError("docker socket unavailable")),
        )

        self.assertEqual(result, 130)
        self.assertEqual(process.wait_calls, 2)


if __name__ == "__main__":
    unittest.main()
