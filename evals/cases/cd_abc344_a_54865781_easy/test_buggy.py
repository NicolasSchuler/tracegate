import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='gpuamkxksz|hkbpphykinkezpl|vfjaqmopod\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'gpuamkxkszvfjaqmopod\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='|krjzrim|lvum\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'lvum\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='renexcfycebeurgvjyo||spdhvuyfvtv\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'renexcfycebeurgvjyospdhvuyfvtv\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='||rdyluacvrayggwnpnzijdifyervjaoal\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'rdyluacvrayggwnpnzijdifyervjaoal\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='xovldqfzaorahdigyojknviaztpcmxl|vovafhjphvshyfiqqtqbx|\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'xovldqfzaorahdigyojknviaztpcmxl\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
