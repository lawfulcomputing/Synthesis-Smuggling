from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(record: dict) -> str:
    plan = record["gemma_private_plan"]
    lines = [
        f"# {record['conversation_id']}",
        "",
        f"Assigned role: {record['assigned_role_gender']}",
        f"Planned reveal turn: {plan['reveal_turn']}",
        f"Scenario: {plan['scenario']}",
        f"Marker trace: {' '.join(record['marker_trace'])}",
        "",
    ]
    for message in record["messages"]:
        if message["role"] == "user":
            lines.append(f"Turn {message['turn']} — Gemma/user: {message['content']}")
        else:
            lines.append(
                f"Turn {message['turn']} — Qwen [{message['suffix_marker']}]: "
                + message["content_visible_to_gemma"]
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the transcripts of a gender simulation for manual review")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--conversation")
    parser.add_argument("--output")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if args.conversation:
        paths = [run_dir / "conversations" / f"{args.conversation}.json"]
    else:
        paths = sorted((run_dir / "conversations").glob("sim_*.json"))
    text = "\n\n---\n\n".join(
        render(json.loads(path.read_text(encoding="utf-8"))) for path in paths
    )
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to replace {output}")
        output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
