import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 2 3\n8 T 4\n7 6 5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 2 3 4 5 6 7\n24 25 26 27 28 29 8\n23 40 41 42 43 30 9\n22 39 48 T 44 31 10\n21 38 47 46 45 32 11\n20 37 36 35 34 33 12\n19 18 17 16 15 14 13\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='9\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 2 3 4 5 6 7 8 9\n32 33 34 35 36 37 38 39 10\n31 56 57 58 59 60 61 40 11\n30 55 72 73 74 75 62 41 12\n29 54 71 80 T 76 63 42 13\n28 53 70 79 78 77 64 43 14\n27 52 69 68 67 66 65 44 15\n26 51 50 49 48 47 46 45 16\n25 24 23 22 21 20 19 18 17\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='11\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 2 3 4 5 6 7 8 9 10 11\n40 41 42 43 44 45 46 47 48 49 12\n39 72 73 74 75 76 77 78 79 50 13\n38 71 96 97 98 99 100 101 80 51 14\n37 70 95 112 113 114 115 102 81 52 15\n36 69 94 111 120 T 116 103 82 53 16\n35 68 93 110 119 118 117 104 83 54 17\n34 67 92 109 108 107 106 105 84 55 18\n33 66 91 90 89 88 87 86 85 56 19\n32 65 64 63 62 61 60 59 58 57 20\n31 30 29 28 27 26 25 24 23 22 21\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='13\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 2 3 4 5 6 7 8 9 10 11 12 13\n48 49 50 51 52 53 54 55 56 57 58 59 14\n47 88 89 90 91 92 93 94 95 96 97 60 15\n46 87 120 121 122 123 124 125 126 127 98 61 16\n45 86 119 144 145 146 147 148 149 128 99 62 17\n44 85 118 143 160 161 162 163 150 129 100 63 18\n43 84 117 142 159 168 T 164 151 130 101 64 19\n42 83 116 141 158 167 166 165 152 131 102 65 20\n41 82 115 140 157 156 155 154 153 132 103 66 21\n40 81 114 139 138 137 136 135 134 133 104 67 22\n39 80 113 112 111 110 109 108 107 106 105 68 23\n38 79 78 77 76 75 74 73 72 71 70 69 24\n37 36 35 34 33 32 31 30 29 28 27 26 25\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
