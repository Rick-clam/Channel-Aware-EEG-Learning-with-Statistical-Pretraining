"""Statically verify the sealed evaluation control-flow invariants."""

import argparse
import ast
import json
from pathlib import Path


def dotted_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def is_not_final_test(node):
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Name)
        and node.operand.id == "is_final_test"
    )


def is_final_only_include_test(node):
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Attribute)
        and dotted_name(node.left) == "args.evaluation_mode"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == "final_test"
    )


def calls(nodes, name):
    found = []
    for node in nodes:
        for descendant in ast.walk(node):
            if isinstance(descendant, ast.Call) and dotted_name(
                descendant.func
            ).endswith(name):
                found.append(descendant)
    return found


def verify_file(path, prepare_names, lock_name, binary=False):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    supervised = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "supervised"
    )
    final_guards = [
        node
        for node in ast.walk(supervised)
        if isinstance(node, ast.If) and is_not_final_test(node.test)
    ]
    fit_guard = next(
        (guard for guard in final_guards if calls(guard.body, "trainer.fit")),
        None,
    )
    test_guard = next(
        (guard for guard in final_guards if calls(guard.orelse, "trainer.test")),
        None,
    )
    if fit_guard is None or test_guard is None:
        raise AssertionError("missing non-final/final evaluation guards")
    fit_calls = calls(fit_guard.body, "trainer.fit")
    test_calls = calls(test_guard.orelse, "trainer.test")
    if len(fit_calls) != 1:
        raise AssertionError("trainer.fit is not confined to the non-final branch")
    if len(test_calls) != 1:
        raise AssertionError("trainer.test is not confined to the final branch")
    if calls(fit_guard.orelse, "trainer.fit"):
        raise AssertionError("final branch contains trainer.fit")
    if calls(test_guard.body, "trainer.test"):
        raise AssertionError("non-final branch contains trainer.test")
    all_fit = calls([supervised], "trainer.fit")
    all_test = calls([supervised], "trainer.test")
    if len(all_fit) != 1 or len(all_test) != 1:
        raise AssertionError("unexpected extra fit/test call in supervised")

    prepare_checks = {}
    for prepare_name in prepare_names:
        call = next(
            (
                call
                for call in calls([supervised], prepare_name)
                if dotted_name(call.func).endswith(prepare_name)
            ),
            None,
        )
        if call is None:
            raise AssertionError(f"missing {prepare_name} call")
        keyword = next(
            (kw for kw in call.keywords if kw.arg == "include_test"), None
        )
        if keyword is None or not is_final_only_include_test(keyword.value):
            raise AssertionError(
                f"{prepare_name} does not restrict test loading to final_test"
            )
        prepare_checks[prepare_name] = True

    lock_calls = calls([supervised], lock_name)
    if len(lock_calls) != 1:
        raise AssertionError("supervised must check exactly one evaluation lock")
    prepare_calls = [
        call
        for name in prepare_names
        for call in calls([supervised], name)
        if dotted_name(call.func).endswith(name)
    ]
    if lock_calls[0].lineno > min(call.lineno for call in prepare_calls):
        raise AssertionError("evaluation lock is checked after a data loader")
    source = path.read_text(encoding="utf-8")
    if 'ckpt_path=checkpoint_record["path"]' not in source:
        raise AssertionError("final test does not name the locked checkpoint")
    if binary and (
        'lightning_model.threshold = evaluation_lock["decision_threshold"]'
        not in source
    ):
        raise AssertionError("binary final test does not load the locked threshold")
    return {
        "file": str(path),
        "fit_calls": len(all_fit),
        "test_calls": len(all_test),
        "fit_only_when_not_final": True,
        "test_only_when_final": True,
        "include_test_only_when_final": prepare_checks,
        "lock_checked_before_loader": True,
        "locked_checkpoint_path_used": True,
        "locked_binary_threshold_used": bool(binary),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = {
        "TUEV": verify_file(
            root / "run_multi_moe.py",
            ["prepare_TUEV_dataloader"],
            "verify_evaluation_lock",
        ),
        "binary": verify_file(
            root / "run_moe.py",
            ["prepare_TUAB_dataloader", "prepare_CHB_MIT_dataloader"],
            "verify_binary_evaluation_lock",
            binary=True,
        ),
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
