from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AddedToken, AutoConfig, AutoTokenizer

from .config import load_config
from .constants import CLASS_NAMES, FEMALE_MARKER, MALE_MARKER
from .data import read_jsonl
from .preprocessing import tokenize_conversation


@dataclass(frozen=True)
class MarkerTokenAudit:
    label: str
    codepoint: str
    token_id: int
    added: bool
    standalone_ids_before: list[int]
    standalone_ids_after: list[int]
    suffix_boundary_stable: bool
    regular_token: bool
    exact_round_trip: bool


def _single_token_round_trip(tokenizer: Any, text: str) -> tuple[bool, list[int]]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return len(ids) == 1 and decoded == text, list(ids)


def ensure_marker_tokens(
    tokenizer: Any,
) -> tuple[dict[str, int], list[MarkerTokenAudit]]:
    marker_ids: dict[str, int] = {}
    audits: list[MarkerTokenAudit] = []
    for label, codepoint, marker in (
        (CLASS_NAMES[0], "U+200B", MALE_MARKER),
        (CLASS_NAMES[1], "U+00A0", FEMALE_MARKER),
    ):
        usable_before, ids_before = _single_token_round_trip(tokenizer, marker)
        added = False
        if not usable_before:
            added_count = tokenizer.add_tokens(
                [
                    AddedToken(
                        marker,
                        single_word=False,
                        lstrip=False,
                        rstrip=False,
                        normalized=False,
                        special=False,
                    )
                ],
                special_tokens=False,
            )
            added = added_count == 1

        usable_after, ids_after = _single_token_round_trip(tokenizer, marker)
        if not usable_after:
            raise ValueError(
                f"Tokenizer cannot represent {codepoint} as one exact token after augmentation: {ids_after}"
            )
        token_id = ids_after[0]
        probe = "marker-boundary-probe"
        probe_ids = tokenizer.encode(probe + marker, add_special_tokens=False)
        probe_decoded = tokenizer.decode(
            probe_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        boundary_stable = (
            bool(probe_ids)
            and probe_ids[-1] == token_id
            and probe_decoded == probe + marker
        )
        if not boundary_stable:
            raise ValueError(
                f"{codepoint} does not remain atomic at an assistant-response suffix"
            )
        regular_token = token_id not in set(tokenizer.all_special_ids)
        if not regular_token:
            raise ValueError(
                f"{codepoint} must be a regular token, not a special token"
            )
        marker_ids[label] = token_id
        audits.append(
            MarkerTokenAudit(
                label=label,
                codepoint=codepoint,
                token_id=token_id,
                added=added,
                standalone_ids_before=ids_before,
                standalone_ids_after=ids_after,
                suffix_boundary_stable=boundary_stable,
                regular_token=regular_token,
                exact_round_trip=usable_after,
            )
        )
    if len(set(marker_ids.values())) != len(marker_ids):
        raise ValueError("The two markers resolved to the same token id")
    return marker_ids, audits


def audit_tokenizer(
    model_name_or_path: str,
    revision: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to replace existing tokenizer audit directory: {output_dir}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, revision=revision, use_fast=True
    )
    model_config = AutoConfig.from_pretrained(model_name_or_path, revision=revision)
    original_tokenizer_size = len(tokenizer)
    marker_ids, audits = ensure_marker_tokens(tokenizer)
    final_tokenizer_size = len(tokenizer)
    embedding_rows = int(model_config.vocab_size)
    max_token_id = max(marker_ids.values())
    if max_token_id >= embedding_rows or final_tokenizer_size > embedding_rows:
        raise ValueError(
            "Tokenizer additions exceed Qwen's embedding capacity; model resizing would be required"
        )

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(tokenizer_dir)
    report = {
        "model_name_or_path": model_name_or_path,
        "requested_revision": revision,
        "resolved_commit": getattr(model_config, "_commit_hash", None),
        "original_tokenizer_size": original_tokenizer_size,
        "final_tokenizer_size": final_tokenizer_size,
        "model_embedding_rows": embedding_rows,
        "unused_embedding_rows_after": embedding_rows - final_tokenizer_size,
        "resizing_required": False,
        "embedding_rows_must_not_be_shrunk": True,
        "marker_token_ids": marker_ids,
        "markers": [asdict(audit) for audit in audits],
    }
    with (output_dir / "tokenizer_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report


def audit_prepared_data(
    tokenizer: Any,
    marker_ids: dict[str, int],
    prepared_dir: str | Path,
    *,
    max_seq_length: int,
    decision_loss_weight: float,
    content_loss_weight: float = 1.0,
) -> dict[str, Any]:
    """Tokenize every prepared record and summarize lengths without model weights."""
    prepared_dir = Path(prepared_dir)
    report: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        path = prepared_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared split for token audit: {path}")
        lengths = []
        for record in read_jsonl(path):
            feature = tokenize_conversation(
                record,
                tokenizer=tokenizer,
                marker_ids=marker_ids,
                max_seq_length=max_seq_length,
                decision_loss_weight=decision_loss_weight,
                content_loss_weight=content_loss_weight,
            )
            lengths.append(len(feature["input_ids"]))
        report[split] = {
            "records": len(lengths),
            "min_tokens": min(lengths),
            "max_tokens": max(lengths),
            "mean_tokens": sum(lengths) / len(lengths),
            "overlength_records": 0,
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit/augment the Qwen tokenizer without loading weights"
    )
    parser.add_argument("--config", default="char_gender/configs/base/qwen2_5_7b_lora.yaml")
    parser.add_argument("--output-dir", default="artifacts/tokenizer_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, project_root = load_config(args.config)
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = project_root / output
    report = audit_tokenizer(
        config["model"]["name_or_path"],
        config["model"]["revision"],
        output,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        output / "tokenizer", use_fast=True, trust_remote_code=False
    )
    marker_ids, _ = ensure_marker_tokens(tokenizer)
    report["prepared_data"] = audit_prepared_data(
        tokenizer,
        marker_ids,
        config["paths"]["prepared_dir"],
        max_seq_length=int(config["training"]["max_seq_length"]),
        decision_loss_weight=float(config["markers"]["decision_loss_weight"]),
        content_loss_weight=float(config["markers"].get("content_loss_weight", 1.0)),
    )
    with (output / "tokenizer_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
