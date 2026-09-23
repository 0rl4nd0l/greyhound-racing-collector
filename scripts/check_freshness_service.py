#!/usr/bin/env python3
"""Execute generated service preflight with kernel-enforced IPv4/IPv6 denial."""
import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def deny_network():
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)

    class Compare(ctypes.Structure):
        _fields_ = [
            ("arg", ctypes.c_uint),
            ("op", ctypes.c_uint),
            ("a", ctypes.c_uint64),
            ("b", ctypes.c_uint64),
        ]

    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(Compare),
    ]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x7FFF0000)
    if not context:
        raise RuntimeError("network_sandbox_unavailable")
    try:
        for domain in (2, 10):
            rule = Compare(0, 4, domain, 0)  # socket(AF_INET/AF_INET6), SCMP_CMP_EQ
            if (
                lib.seccomp_rule_add_array(
                    context,
                    0x50000 | errno.EPERM,
                    lib.seccomp_syscall_resolve_name(b"socket"),
                    1,
                    ctypes.byref(rule),
                )
                != 0
            ):
                raise RuntimeError("network_sandbox_rule_failed")
        if lib.seccomp_load(context) != 0:
            raise RuntimeError("network_sandbox_load_failed")
    finally:
        lib.seccomp_release(context)


def service_command(unit):
    lines = Path(unit).read_text().splitlines()
    command = shlex.split(
        next(line.split("=", 1)[1] for line in lines if line.startswith("ExecStart="))
    )
    cwd = next(line.split("=", 1)[1] for line in lines if line.startswith("WorkingDirectory="))
    environment = {"HOME": str(Path.home()), "PYTHONDONTWRITEBYTECODE": "1"}
    for line in lines:
        if line.startswith("Environment="):
            for assignment in shlex.split(line.split("=", 1)[1]):
                key, value = assignment.split("=", 1)
                environment[key] = value
    return command, cwd, environment


def check(unit):
    command, cwd, environment = service_command(unit)
    deny_network()
    completed = subprocess.run(
        [*command, "--verify-live-runtime"],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode:
        raise ValueError("generated_service_preflight_failed: " + completed.stderr[-3000:])
    capture_identity = json.loads(completed.stdout.splitlines()[-1])
    versions = {}
    import re

    for name, binary in capture_identity["browser_binaries"].items():
        version = subprocess.run(
            [binary["path"], "--version"],
            env=environment,
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        )
        versions[name] = version.stdout.strip()
    majors = [re.search(r"\b(\d+)\.", version).group(1) for version in versions.values()]
    if len(set(majors)) != 1:
        raise ValueError("installed_browser_driver_version_mismatch")
    return {
        "browser_versions": versions,
        "command": command,
        "cwd": cwd,
        "network": "KERNEL_IPV4_IPV6_DENIED",
        "capture_child": capture_identity,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=Path, required=True)
    print(json.dumps(check(parser.parse_args().unit), sort_keys=True))
