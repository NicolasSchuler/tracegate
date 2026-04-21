import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n0 5 1 3\n1 4 0 5\n2 5 2 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '20\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n0 100 0 100\n0 100 0 100\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n0 1 0 1\n0 3 0 5\n5 10 0 10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '65\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n0 100 0 100\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n26 76 26 76\n20 70 20 70\n40 90 40 90\n12 62 38 88\n28 78 28 78\n33 83 33 83\n1 51 49 99\n38 88 12 62\n17 67 17 67\n23 73 27 77\n19 69 19 69\n49 99 1 51\n30 80 20 70\n26 76 24 74\n22 72 22 72\n11 61 39 89\n32 82 18 68\n22 72 28 78\n35 85 15 65\n10 60 40 90\n31 81 19 69\n9 59 41 91\n3 53 3 53\n8 58 42 92\n46 96 46 96\n34 84 16 66\n28 78 22 72\n1 51 1 51\n24 74 26 76\n13 63 37 87\n4 54 4 54\n4 54 46 96\n30 80 30 80\n2 52 48 98\n32 82 32 82\n3 53 47 97\n44 94 6 56\n11 61 11 61\n31 81 31 81\n21 71 21 71\n35 85 35 85\n5 55 5 55\n49 99 49 99\n40 90 10 60\n48 98 48 98\n27 77 27 77\n7 57 43 93\n14 64 36 86\n8 58 8 58\n45 95 5 55\n16 66 34 84\n43 93 43 93\n27 77 23 73\n15 65 35 85\n41 91 9 59\n45 95 45 95\n36 86 14 64\n7 57 7 57\n38 88 38 88\n39 89 39 89\n46 96 4 54\n18 68 18 68\n17 67 33 83\n6 56 6 56\n42 92 42 92\n25 75 25 75\n33 83 17 67\n34 84 34 84\n15 65 15 65\n41 91 41 91\n12 62 12 62\n48 98 2 52\n18 68 32 82\n23 73 23 73\n36 86 36 86\n2 52 2 52\n50 100 0 50\n5 55 45 95\n29 79 21 71\n10 60 10 60\n29 79 29 79\n39 89 11 61\n6 56 44 94\n9 59 9 59\n25 75 25 75\n43 93 7 57\n0 50 0 50\n42 92 8 58\n16 66 16 66\n44 94 44 94\n13 63 13 63\n47 97 47 97\n47 97 3 53\n37 87 13 63\n19 69 31 81\n37 87 37 87\n20 70 30 80\n24 74 24 74\n14 64 14 64\n21 71 29 79\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '9802\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
