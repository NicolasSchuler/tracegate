"""Tests for baseline repo normalization after patch application."""

from __future__ import annotations

from pathlib import Path

from baselines.normalize_repos import normalize_baseline_repos


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_normalize_baseline_repos_repairs_known_ldb_and_tracecoder_merges(
    tmp_path: Path,
) -> None:
    ldb_model = tmp_path / "LLMDebugger" / "programming" / "generators" / "model.py"
    ldb_driver = tmp_path / "LLMDebugger" / "programming" / "ldb.py"
    ldb_utils = tmp_path / "LLMDebugger" / "programming" / "utils.py"
    trace_entry = tmp_path / "TraceCoder" / "trace_learn_coder.py"
    _write(
        ldb_model,
        (
            "from openai import OpenAIfrom transformers import GPT2Tokenizer, AutoTokenizer\n"
            "retry,    stop_after_attempt,  # type: ignore\n"
            "model: str,    prompt: str,\n"
            "role: MessageRole    content: str\n"
            'def __init__(self, model_name: str, key: str = ""):        self.name = model_name\n'
            'self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")        if key != "":\n'
            "messages = new_messages            response = self.client.chat.completions.create(\n"
            "temperature=temperature,                    top_p=1,\n"
            "try:                responses = self.vllm_client.completions.create(\n"
            "res = self.gpt_chat(messages, stop, max_tokens, temperature, num_comps)        return res\n"
            'if "maximum context length" in str(e):                    max_length -= 2000\n'
        ),
    )
    _write(
        ldb_driver,
        (
            "model = model_factory(model_name, port)    cur_pass = 0\n"
            "tests_i = item['given_tests']        # clean test_i\n"
            'last_func_impl = ""        if model.is_chat:\n'
            'debug_cur_func_impl = insert_comment(cur_func_impl, extrace_comment(item["prompt"]), item["entry_point"])                else:\n'
            "model=model,                prev_func_impl=cur_func_impl,\n"
            "if isinstance(messages, str):                token_nums += len(tokenizer.tokenize(messages))\n"
            "if is_passing or cur_iter == max_iters - 1:                if is_passing:\n"
            "cur_iter += 1                sys.stdout.flush()\n"
            'item["is_solved"] = is_solved    item["implementations"] = implementations\n'
        ),
    )
    _write(
        ldb_utils,
        "cur_func_impl = seed        else:\n",
    )
    _write(
        trace_entry,
        (
            "EXPERIMENT_ID = f\"{args.dataset}_{args.model.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}\"    OUTPUT_CSV_FILENAME = args.output_file or f\"results/experiment_{EXPERIMENT_ID}.csv\"\n"
            'print(f"Error loading dataset: {e}")        sys.exit(1)\n'
            'sys.exit(1)    except KeyError: # Added KeyError catch in case the "data_path" key is missing\n'
            'def main():    """\n'
            "problem_keys = list(datasets.keys())    start_index = args.start_index\n"
            "if args.dataset not in DATASET_PATHS:        print(\n"
        ),
    )

    changed = normalize_baseline_repos(tmp_path)

    assert "LLMDebugger/programming/generators/model.py" in changed
    assert "LLMDebugger/programming/ldb.py" in changed
    assert "LLMDebugger/programming/utils.py" in changed
    assert "TraceCoder/trace_learn_coder.py" in changed

    ldb_text = ldb_model.read_text(encoding="utf-8")
    assert "OpenAIfrom transformers" not in ldb_text
    assert "retry,    stop_after_attempt" not in ldb_text
    assert "model: str,    prompt: str," not in ldb_text
    assert "role: MessageRole    content: str" not in ldb_text
    assert 'key: str = ""):        self.name' not in ldb_text
    assert 'from_pretrained("gpt2")        if key != "":' not in ldb_text
    assert "new_messages            response" not in ldb_text
    assert "temperature=temperature,                    top_p=1," not in ldb_text
    assert "try:                responses" not in ldb_text
    assert "num_comps)        return res" not in ldb_text
    assert (
        '"maximum context length" in str(e):                    max_length -= 2000' not in ldb_text
    )
    ldb_driver_text = ldb_driver.read_text(encoding="utf-8")
    assert "port)    cur_pass" not in ldb_driver_text
    assert "given_tests']        # clean" not in ldb_driver_text
    assert '""        if model.is_chat' not in ldb_driver_text
    assert 'item["entry_point"])                else:' not in ldb_driver_text
    assert "model=model,                prev_func_impl" not in ldb_driver_text
    assert "str):                token_nums" not in ldb_driver_text
    assert "1:                if is_passing" not in ldb_driver_text
    assert "1                sys.stdout.flush()" not in ldb_driver_text
    assert 'is_solved"] = is_solved    item["implementations"]' not in ldb_driver_text
    ldb_utils_text = ldb_utils.read_text(encoding="utf-8")
    assert "seed        else:" not in ldb_utils_text

    trace_text = trace_entry.read_text(encoding="utf-8")
    assert "OUTPUT_CSV_FILENAME" in trace_text
    assert '"{e}")        sys.exit(1)' not in trace_text
    assert "sys.exit(1)    except KeyError:" not in trace_text
    assert 'def main():    """' not in trace_text
    assert "keys())    start_index" not in trace_text
    assert "DATASET_PATHS:        print(" not in trace_text


def test_normalize_baseline_repos_is_idempotent_on_clean_files(tmp_path: Path) -> None:
    clean = tmp_path / "TraceCoder" / "trace_learn_coder.py"
    _write(clean, "problem_keys = list(datasets.keys())\n    start_index = args.start_index\n")

    first = normalize_baseline_repos(tmp_path)
    second = normalize_baseline_repos(tmp_path)

    assert first == []
    assert second == []
