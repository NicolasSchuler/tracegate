import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 4\n.1.#\n###.\n.#2.\n#.##\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '...#\n#...\n....\n#...\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 5\n..#.#\n###.#\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '..#.#\n###.#\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 3\n11#\n###\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '...\n..#\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 6\n#.#3#.\n###.#.\n##.###\n#1..#.\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '......\n#.....\n#....#\n....#.\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='16 19\n#.#....#..##.####..\n.#..######....#.#.#\n.####..##....#.#.#.\n#...#..#.##...####.\n.#######.#...##....\n.####.##.#..###...#\n..#.###.##..#.##...\n.#.##..###..###.#.#\n#####.#.#...#.####.\n.#..##....#...#..#.\n.###.#..#..###....#\n..##.#.#.##..##....\n..#.####.##.##..#..\n.#..####..#.##..#.#\n.##..#...##..#.#.##\n..#....#.#.#.#.#.##\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '#.#....#..##.####..\n.#..######....#.#.#\n.####..##....#.#.#.\n#...#..#.##...####.\n.#######.#...##....\n.####.##.#..###...#\n..#.###.##..#.##...\n.#.##..###..###.#.#\n#####.#.#...#.####.\n.#..##....#...#..#.\n.###.#..#..###....#\n..##.#.#.##..##....\n..#.####.##.##..#..\n.#..####..#.##..#.#\n.##..#...##..#.#.##\n..#....#.#.#.#.#.##\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
