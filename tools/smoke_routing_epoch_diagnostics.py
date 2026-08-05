"""Exercise complete held-out routing diagnostics without a Trainer."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.dwmoespace_newgate import dwmoespace_newgate
from run_multi_moe import LitModel_finetune


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def main():
    torch.manual_seed(2026)
    model = dwmoespace_newgate(
        6,
        gate_type="input",
        expert_axis="temporal",
        norm_type="local_filter",
        relation_mode="dynamic_normalized",
    )
    lightning = LitModel_finetune(
        SimpleNamespace(seed=2026, lr=1e-3, weight_decay=1e-5),
        model,
    )
    # test_epoch_end logs through Lightning in production; this standalone
    # smoke replaces logging while leaving the diagnostic computation intact.
    lightning.log = lambda *args, **kwargs: None
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    outputs = [
        lightning.test_step((torch.randn(6, 16, 2000), labels), index)
        for index in range(2)
    ]
    metrics = lightning.test_epoch_end(outputs)
    report = {
        "metrics": metrics,
        "num_samples": int(lightning.test_labels.shape[0]),
        "expert_logits_shape": list(lightning.test_expert_logits.shape),
        "gate_weights_shape": list(lightning.test_gate_weights.shape),
        "permuted_gate_logits_shape": list(
            lightning.test_permuted_gate_logits.shape
        ),
        "permutation_shape": list(lightning.test_gate_permutation.shape),
        "permutation_is_bijection": bool(
            np.array_equal(
                np.sort(lightning.test_gate_permutation),
                np.arange(lightning.test_labels.shape[0]),
            )
        ),
        "causal_routing": lightning.test_causal_diagnostics,
    }
    print(json.dumps(report, indent=2, default=json_default))
    if report["num_samples"] != 12:
        raise SystemExit("Test batches were not concatenated globally")
    if not report["permutation_is_bijection"]:
        raise SystemExit("Gate permutation is not a full-test bijection")
    if report["permuted_gate_logits_shape"] != [12, 6]:
        raise SystemExit("Unexpected permuted-logit shape")


if __name__ == "__main__":
    main()
