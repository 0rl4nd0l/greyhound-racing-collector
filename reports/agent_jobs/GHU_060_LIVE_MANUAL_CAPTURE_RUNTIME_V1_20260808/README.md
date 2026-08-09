# GHU-060 implementation evidence

This report covers the GHU-060 corrective network-policy pass on PR #128,
starting from exact PR head `8bcb9cd574549871b5f6de71edd4a62e4a2a0cd7` and
base `0a67f95ea06effa04609faabe0103fe2e69ff94e`.

The child now permits only the exact canonical race document navigation and
reviewed query-free static assets under the trusted `/assets/` path. XHR,
fetch, websocket, event-stream, API, result-like, unknown, and unclassified
requests fail closed. No reviewed odds API endpoint was found, so none is
allowed.

The implementation is fixture-tested only. No live browser/source attempt,
deployment, installation, activation, service restart, lock manipulation,
canonical/Phase-7 write, or merge occurred.
