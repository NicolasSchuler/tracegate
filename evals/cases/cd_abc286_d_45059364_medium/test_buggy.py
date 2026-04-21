import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 19\n2 3\n5 6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 18\n2 3\n5 6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 1001\n1 1\n2 1\n100 10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='50 10000\n76 50\n85 50\n61 50\n93 50\n67 50\n78 50\n52 50\n82 50\n94 50\n59 50\n98 50\n99 50\n56 50\n86 50\n88 50\n58 50\n92 50\n65 50\n80 50\n77 50\n87 50\n97 50\n62 50\n66 50\n51 50\n79 50\n54 50\n71 50\n63 50\n73 50\n72 50\n74 50\n70 50\n68 50\n55 50\n95 50\n100 50\n53 50\n96 50\n81 50\n60 50\n91 50\n83 50\n84 50\n57 50\n89 50\n90 50\n75 50\n64 50\n69 50\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='50 7551\n63 2\n52 2\n86 2\n75 2\n64 2\n70 2\n84 2\n73 2\n85 2\n71 2\n53 2\n51 2\n69 2\n83 2\n91 2\n68 2\n57 2\n60 2\n72 2\n58 2\n97 2\n95 2\n93 2\n87 2\n98 2\n65 2\n79 2\n90 2\n62 2\n89 2\n77 2\n92 2\n54 2\n94 2\n80 2\n67 2\n55 2\n81 2\n78 2\n61 2\n56 2\n82 2\n59 2\n100 2\n74 2\n96 2\n99 2\n88 2\n76 2\n66 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
