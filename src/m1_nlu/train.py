from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.m1_nlu.dataset import category_distribution, load_samples, split_samples, write_split_manifest
from src.m1_nlu.model import JointNLUConfig, JointNLUModel


def train_m1(
    train_file: str,
    out_file: str,
    model_path: str = "artifacts/m1/model.json",
) -> dict:
    samples = load_samples(train_file)
    train, val, test = split_samples(samples)
    train_dist = category_distribution(train)
    val_dist = category_distribution(val)
    test_dist = category_distribution(test)

    config = JointNLUConfig(num_intents=max(1, len(train_dist)), num_slot_labels=16)
    model = JointNLUModel(config)
    model.fit(train)

    artifact = {
        "config": {
            "num_intents": config.num_intents,
            "num_slot_labels": config.num_slot_labels,
            "backbone": config.backbone,
        },
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "train_distribution": train_dist,
        "val_distribution": val_dist,
        "test_distribution": test_dist,
        "label_priors": model.label_priors,
        "model_type": "multinomial_nb_intent + token_tag_slot",
    }

    p = Path(out_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    write_split_manifest(train, val, test, "data/annotations/m1_split_manifest.json")

    model_dump = {
        "config": artifact["config"],
        "label_priors": model.label_priors,
        "intent_log_priors": model.intent_log_priors,
        "intent_log_likelihoods": model.intent_log_likelihoods,
        "default_log_prob": model.default_log_prob,
        "slot_token_tag": model.slot_token_tag,
    }
    model_out = Path(model_path)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.write_text(json.dumps(model_dump, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/annotations/m1_train.json")
    parser.add_argument("--out_file", default="artifacts/m1/model_stub.json")
    args = parser.parse_args()
    out = train_m1(args.train_file, args.out_file)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
