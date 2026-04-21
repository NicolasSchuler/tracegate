"""Normalize known git-apply formatting artifacts in cloned baseline repos.

Some upstream compatibility patches pick up line-collapsing artifacts under
different git environments. These replacements are intentionally narrow and
idempotent: they only repair known broken token sequences produced by fresh
patch application, and they leave already-correct checkouts unchanged.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileNormalization:
    repo_name: str
    relative_path: str
    replacements: tuple[tuple[str, str], ...]


_NORMALIZATIONS = (
    FileNormalization(
        repo_name="LLMDebugger",
        relative_path="programming/generators/model.py",
        replacements=(
            (
                "from openai import OpenAIfrom transformers import GPT2Tokenizer, AutoTokenizer",
                "from openai import OpenAI\nfrom transformers import GPT2Tokenizer, AutoTokenizer",
            ),
            (
                "retry,    stop_after_attempt,  # type: ignore",
                "retry,\n    stop_after_attempt,  # type: ignore",
            ),
            (
                "model: str,    prompt: str,",
                "model: str,\n    prompt: str,",
            ),
            (
                "role: MessageRole    content: str",
                "role: MessageRole\n    content: str",
            ),
            (
                'def __init__(self, model_name: str, key: str = ""):        self.name = model_name',
                'def __init__(self, model_name: str, key: str = ""):\n        self.name = model_name',
            ),
            (
                'self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")        if key != "":',
                'self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")\n        if key != "":',
            ),
            (
                "messages = new_messages            response = self.client.chat.completions.create(",
                "messages = new_messages\n            response = self.client.chat.completions.create(",
            ),
            (
                "temperature=temperature,                    top_p=1,",
                "temperature=temperature,\n                    top_p=1,",
            ),
            (
                "try:                responses = self.vllm_client.completions.create(",
                "try:\n                responses = self.vllm_client.completions.create(",
            ),
            (
                "res = self.gpt_chat(messages, stop, max_tokens, temperature, num_comps)        return res",
                "res = self.gpt_chat(messages, stop, max_tokens, temperature, num_comps)\n        return res",
            ),
            (
                'if "maximum context length" in str(e):                    max_length -= 2000',
                'if "maximum context length" in str(e):\n                    max_length -= 2000',
            ),
        ),
    ),
    FileNormalization(
        repo_name="LLMDebugger",
        relative_path="programming/ldb.py",
        replacements=(
            (
                "model = model_factory(model_name, port)    cur_pass = 0",
                "model = model_factory(model_name, port)\n    cur_pass = 0",
            ),
            (
                "tests_i = item['given_tests']        # clean test_i",
                "tests_i = item['given_tests']\n        # clean test_i",
            ),
            (
                'last_func_impl = ""        if model.is_chat:',
                'last_func_impl = ""\n        if model.is_chat:',
            ),
            (
                'debug_cur_func_impl = insert_comment(cur_func_impl, extrace_comment(item["prompt"]), item["entry_point"])                else:',
                'debug_cur_func_impl = insert_comment(cur_func_impl, extrace_comment(item["prompt"]), item["entry_point"])\n                else:',
            ),
            (
                "model=model,                prev_func_impl=cur_func_impl,",
                "model=model,\n                prev_func_impl=cur_func_impl,",
            ),
            (
                "if isinstance(messages, str):                token_nums += len(tokenizer.tokenize(messages))",
                "if isinstance(messages, str):\n                token_nums += len(tokenizer.tokenize(messages))",
            ),
            (
                "if is_passing or cur_iter == max_iters - 1:                if is_passing:",
                "if is_passing or cur_iter == max_iters - 1:\n                if is_passing:",
            ),
            (
                "cur_iter += 1                sys.stdout.flush()",
                "cur_iter += 1\n                sys.stdout.flush()",
            ),
            (
                'item["is_solved"] = is_solved    item["implementations"] = implementations',
                'item["is_solved"] = is_solved\n    item["implementations"] = implementations',
            ),
        ),
    ),
    FileNormalization(
        repo_name="LLMDebugger",
        relative_path="programming/utils.py",
        replacements=(
            (
                "cur_func_impl = seed        else:",
                "cur_func_impl = seed\n        else:",
            ),
        ),
    ),
    FileNormalization(
        repo_name="TraceCoder",
        relative_path="trace_learn_coder.py",
        replacements=(
            (
                "EXPERIMENT_ID = f\"{args.dataset}_{args.model.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}\"    OUTPUT_CSV_FILENAME = args.output_file or f\"results/experiment_{EXPERIMENT_ID}.csv\"",
                "EXPERIMENT_ID = f\"{args.dataset}_{args.model.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}\"\n    OUTPUT_CSV_FILENAME = args.output_file or f\"results/experiment_{EXPERIMENT_ID}.csv\"",
            ),
            (
                'print(f"Error loading dataset: {e}")        sys.exit(1)',
                'print(f"Error loading dataset: {e}")\n        sys.exit(1)',
            ),
            (
                'sys.exit(1)    except KeyError: # Added KeyError catch in case the "data_path" key is missing',
                'sys.exit(1)\n    except KeyError: # Added KeyError catch in case the "data_path" key is missing',
            ),
            (
                'def main():    """',
                'def main():\n    """',
            ),
            (
                "problem_keys = list(datasets.keys())    start_index = args.start_index",
                "problem_keys = list(datasets.keys())\n    start_index = args.start_index",
            ),
            (
                "if args.dataset not in DATASET_PATHS:        print(",
                "if args.dataset not in DATASET_PATHS:\n        print(",
            ),
        ),
    ),
)


def normalize_baseline_repos(root: Path) -> list[str]:
    """Apply idempotent textual repairs to known fresh-apply formatting artifacts."""
    resolved_root = root.resolve()
    changed_files: list[str] = []
    for spec in _NORMALIZATIONS:
        path = resolved_root / spec.repo_name / spec.relative_path
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        original = text
        for before, after in spec.replacements:
            text = text.replace(before, after)
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed_files.append(str(path.relative_to(resolved_root)))
    return changed_files


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0]) if args else Path(__file__).resolve().parent / "repos"
    changed = normalize_baseline_repos(root)
    for relative_path in changed:
        print(f"[baseline-normalize] repaired {relative_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
