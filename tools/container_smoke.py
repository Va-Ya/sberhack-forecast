"""Run inside the offline container; POST supplied CSVs over its loopback interface."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/inputs")
    parser.add_argument("--network-mode", choices=["none", "host", "unspecified"], default="unspecified",
                        help="Declared environment; use none only with docker run --network none")
    args = parser.parse_args()
    directory = Path(args.input_dir)
    deadline = time.monotonic() + 60
    while True:
        connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=3)
        try:
            connection.request("GET", "/health")
            health = connection.getresponse()
            assert health.status == 200
            health.read()
            connection.close()
            break
        except (OSError, http.client.HTTPException):
            connection.close()
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=3600)
    boundary = "SberHackOfflineBoundary2026"
    parts = []
    total = 0
    for kind in ["sales", "stock"]:
        header = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{kind}\"; "
                  f"filename=\"{kind}.csv\"\r\nContent-Type: text/csv\r\n\r\n").encode()
        path = directory / f"{kind}.csv"
        parts.append((header, path))
        total += len(header) + path.stat().st_size + 2
    tail = f"--{boundary}--\r\n".encode()
    total += len(tail)
    started = time.perf_counter()
    connection.putrequest("POST", "/predict")
    connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
    connection.putheader("Content-Length", str(total))
    connection.endheaders()
    for header, path in parts:
        connection.send(header)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                connection.send(chunk)
        connection.send(b"\r\n")
    connection.send(tail)
    response = connection.getresponse()
    body = response.read()
    assert response.status == 200, body[:1000]
    expected = (directory / "forecast.csv").read_bytes()
    # CSV output is stable across platforms; accept only numerical roundoff up to 1e-7.
    import io

    import numpy as np
    import pandas as pd

    actual = pd.read_csv(io.BytesIO(body))
    reference = pd.read_csv(io.BytesIO(expected))
    pd.testing.assert_frame_equal(actual.drop(columns="quantity"), reference.drop(columns="quantity"))
    np.testing.assert_allclose(actual.quantity, reference.quantity, rtol=0, atol=1e-7)
    # Parsing/importing pandas can exceed uvicorn's keep-alive timeout on a small host.
    connection.close()
    connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=600)
    connection.request("GET", "/predict/all?format=csv")
    cached = connection.getresponse()
    assert cached.status == 200 and cached.read() == body
    print(json.dumps({"status": "passed", "network": args.network_mode, "rows": len(actual),
                       "seconds": time.perf_counter() - started, "sha256": hashlib.sha256(body).hexdigest(),
                       "exact_bytes_equal": body == expected}), flush=True)


if __name__ == "__main__":
    main()
