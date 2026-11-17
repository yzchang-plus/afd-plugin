#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Elastic AFD validation helper: serve probes and resize calls.

Standard library only, so it runs in any CANN/vLLM environment without
checking whether `requests` happens to be installed.

Subcommands:
  probe   send fixed prompts at temperature=0 and hash the returned texts
  scale   POST /scale_elastic_ep and map the HTTP status to an exit code

Exit codes:
  probe   0 ok, 1 transport/parse failure
  scale   0 status matched --expect, 2 status was valid but unexpected
          (i.e. a 400 where a 200 was required or vice versa), 1 other failure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request

DEFAULT_PROMPTS = (
    "Explain the difference between attention and FFN disaggregation.",
    "The capital of France is",
    "def fibonacci(n):",
)


def _post(url: str, payload: dict, timeout: float) -> tuple[int, str]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode(errors="replace")
    except urllib.error.URLError as exc:
        return -1, f"transport failure: {exc.reason}"


def prompts(path: str | None) -> list[str]:
    if not path:
        return list(DEFAULT_PROMPTS)
    with open(path, encoding="utf-8") as handle:
        return [line for line in (raw.strip() for raw in handle) if line]


def cmd_probe(args: argparse.Namespace) -> int:
    items = prompts(args.prompts_file)
    texts: list[str] = []
    for prompt in items:
        status, raw = _post(
            f"{args.base_url}/v1/completions",
            {
                "model": args.model,
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": args.max_tokens,
            },
            args.timeout,
        )
        if status != 200:
            print(f"PROBE_FAIL status={status} body={raw[:400]}", file=sys.stderr)
            return 1
        try:
            decoded = json.loads(raw)
            texts.append(str(decoded["choices"][0]["text"]))
        except (KeyError, IndexError, ValueError) as exc:
            print(f"PROBE_FAIL parse={exc!r} body={raw[:400]}", file=sys.stderr)
            return 1
    digest = hashlib.sha256("\n---\n".join(texts).encode()).hexdigest()
    if args.save:
        with open(args.save, "w", encoding="utf-8") as handle:
            handle.write(digest + "\n")
            for text in texts:
                handle.write(text + "\n---\n")
    print(f"PROBE_OK digest={digest} prompts={len(items)}")
    return 0


def cmd_scale(args: argparse.Namespace) -> int:
    payload = {
        "new_data_parallel_size": args.new_size,
        "role": args.role,
        "drain_timeout": args.drain_timeout,
    }
    status, raw = _post(
        f"{args.base_url}/scale_elastic_ep", payload, args.timeout + args.drain_timeout
    )
    print(f"SCALE role={args.role} new_size={args.new_size} status={status}")
    if status != args.expect:
        print(f"SCALE_MISMATCH expected={args.expect} got={status} body={raw[:600]}")
        return 1 if status == -1 else 2
    if status == 200:
        print("SCALE_OK")
        return 0
    # A rejected request must fail cleanly: the reason belongs in the log.
    print(f"SCALE_REJECTED as expected body={raw[:600]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120.0)
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe")
    p_probe.add_argument("--model", required=True)
    p_probe.add_argument("--max-tokens", type=int, default=32)
    p_probe.add_argument("--prompts-file", default=None)
    p_probe.add_argument("--save", default=None)
    p_probe.set_defaults(func=cmd_probe)

    p_scale = sub.add_parser("scale")
    p_scale.add_argument("--role", required=True, choices=("attention", "ffn"))
    p_scale.add_argument("--new-size", type=int, required=True)
    p_scale.add_argument("--drain-timeout", type=int, default=120)
    p_scale.add_argument("--expect", type=int, default=200, help="required HTTP status")
    p_scale.set_defaults(func=cmd_scale)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
