---
name: docker-patterns
description: Use for packaging and verifying the forecast service.
license: MIT
---

# docker-patterns

Use for packaging and verifying the forecast service.
Use a pinned Python slim image, pinned packages, libgomp1 for LightGBM and a non-root runtime user.
Copy only forecast, app and the model bundle; exclude training data and archives.
Use one uvicorn worker, health checks, bounded CPU threads and no inference-time downloads.
Verify with a network-disabled container and input files provided externally.
See references/upstream.md for optional container patterns; this project needs no database or Redis.

Source: https://github.com/Shelpess/komanda/tree/c8dfa617a2ef1523c713ba79e88189d93b0f1173/resources/ecc/upstream/skills/docker-patterns
