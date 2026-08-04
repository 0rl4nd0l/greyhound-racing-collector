# Operator UI R3 generated deployment

Status: GHU-036 review candidate. This is not an installed, accepted, deployed,
or runtime-proven service.

## Boundary and package

`bin/operator-ui-deployment generate` is the only repository-owned generator.
It emits `greyhound-operator-ui-r3.service`, `operator-ui-r3.env`, and
`ROLLBACK.md`, and writes the fixed
`var/operator_ui/generated/repository-v1.binding.json` consumed by
`src/operator_ui/bootstrap.py`. Do not hand-edit an installed or generated
copy. Regenerate it from the exact reviewed source instead.

The generator requires exact source commit/tree/version/profile identities; a
pinned regular owner-executable Python; the checked-in repository-v1 profile
and five fixed prediction artifacts; the authoritative canonical database,
evidence/current-index and collector-protocol roots; the authoritative
prediction-bundle producer root; and a separate writable Operator UI operations
root. Missing, symlinked, unsafe, overlapping, malformed, or public inputs fail
before package output is written. The service namespace mounts source,
collector evidence, producer evidence, and the canonical database read-only;
only the separate operations root is writable.

An enabled package additionally requires `--live-authority` pointing to one
server-owned `operator_ui_live_authority_v1` observation. It names the exact
full/odds state and report files, the odds report's same-run refresh file, the
inventory report/manifest and all seven raw packet files, the predictor catalog
and exact config/schema/model/manifest bytes, four installed unit files, and
the observed `ActiveState`/`SubState`/`ExecMainPID` values. The generator copies
no evidence: it performs bounded no-follow retained reads and seals each fixed
path and byte identity in the repository binding. `observed_at` must be the
actual time those installed-unit and service-state observations were captured;
the generator does not invent it. Regenerate after any authoritative producer
or installed-unit observation changes. Browser requests and environment values
cannot select paths or commands.

The default package is disabled. Omitting `--enable` writes
`OPERATOR_UI_CONNECTED_MODE=0`, `OPERATOR_UI_LEVEL=1`, and
`OPERATOR_UI_R3_PROFILE=disabled`; its guarded
entrypoint exits without starting another UI process. `--enable` is only a
package-generation choice; it writes `OPERATOR_UI_LEVEL=2` and requires the
complete live authority observation, but is not permission to install or start a service.
The separate unit does not replace the existing UI, which remains available
until an accepted deployment deliberately changes that state.

## Bind and access

The default bind is `127.0.0.1:5055`. The generator accepts only an IP address
classified as loopback or private and rejects wildcard, multicast, and public
addresses. Prefer loopback with a separately approved local/Tailscale access
path. Tailscale/private reachability is not authentication: connected-mode
login, HTTPS termination/access controls, and the private exposure review are
still required. Do not port-forward or publish this development server.

## Secrets

Create the secrets file outside the repository and generated package. It must
be a pre-existing regular non-symlink file with mode `0600`, containing exactly
one assignment for each of `OPERATOR_UI_SECRET_KEY`,
`OPERATOR_UI_USERNAME`, and a Werkzeug scrypt/pbkdf2
`OPERATOR_UI_PASSWORD_HASH`, with no other assignments. The generator validates
names and permissions but never copies secret values into the service,
environment file, binding, or documentation. The generated unit references the
original file with `EnvironmentFile=`. Never commit the file or paste its
contents into review evidence.

## Enable and verify (after separate Level 4 approval)

Generate in a new empty, owner-only output directory using all explicit
absolute authority paths and the exact reviewed `--source-commit` and
`--source-tree`. Add `--enable` only for the approved enabled candidate. Review
the four generated files and their hashes. Installation and service lifecycle
actions are intentionally outside this generator.

Before any start, verify the unit with `systemd-analyze verify`, confirm the
unit/config/binding hashes equal the accepted review packet, confirm the bind
is the approved private address, and confirm the deployed source commit/tree,
profile hash, five artifact hashes, Python, database, evidence, producer, and
operations paths exactly match the binding. After an authorized start, inspect
the installed unit and process identity, confirm there is no wildcard/public
listener, authenticate over the approved access path, verify the existing UI
is still available, and inspect only Level-1/Level-2 read/audit behavior. This
ticket authorizes no prediction, collector/browser action, or live proof.

## Rollback

Regenerate without `--enable`, then—under separate service authority—stop and
disable only `greyhound-operator-ui-r3.service` and verify the existing UI.
Rollback disables the feature; it does not delete or rewrite the operations
root, audit/job databases, prediction bundles, collector evidence, current
index, protocol evidence, or canonical database. Preserve the rejected or
rolled-back package and hashes as review evidence. Never remove evidence as a
rollback shortcut.
