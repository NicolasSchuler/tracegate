import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='abcdaxayz\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '7\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='bcbbbz\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='aaaaa\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='lpixtnpukwdrlfjzkipzdqxiptrmkwtyqazyifpwpwqhebycyrnsnmwfqgghhemrqbxcmrwfqyyhczksgblxasbiqkrpniybvevw\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '85\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='ueqonvbdgvodlguqnbuqklnvrlfbhogwkrdmxwiqhscpmxxnnnlvoribrpkvnsqmvpcqrotzzgidiebptnryogxodezjwjdhofgs\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
