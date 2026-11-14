from __future__ import annotations

import argparse
import json

from .config import load_config
from .data import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build marker-augmented conversational JSONL splits"
    )
    parser.add_argument("--config", default="char_gender/configs/base/qwen2_5_7b_lora.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, _ = load_config(args.config)
    manifest = prepare_dataset(
        config["paths"]["source_data"],
        config["paths"]["prepared_dir"],
        expected_totals=config["dataset"]["expected_label_counts"],
        expected_split_sizes=config["dataset"]["expected_split_counts"],
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
