import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n3 5 10\n4 3 3\n2 2 6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '17\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n3 5 10\n4 3 3\n2 2 3\n',
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
        input='2\n4 8\n3 1 100\n4 10000 100\n',
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
        input='99\n14575 36426 79445 48772 90081 33447 90629 3497 47202 7775 94325 63982 4784 68417 2156 31932 35902 95728 78537 23857 30739 86918 29211 39679 38506 63340 86568 61868 60016 87940 96263 24593 1449 36991 90310 23355 77068 11431 8580 91757 49218 74934 94328 63676 29355 96221 99080 95922 31545 48511 34067 75467 38674 84691 6504 99835 62034 44965 59980 81221 45895 32501 48152 8325 47731 29302 48153 9896 19022 97489 8399 70607 66466 74432 31502 53968 28333 87475 33792 84 87010 79694 27328 7354 24712 38409 16480 2643 28121 51951 23492 4420 28197 28607 97925 72167 60717 63438 89200\n99925 277429832 713\n100000 277429834 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '27465553368\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='97\n78720 87658 90639 89992 49130 26467 98989 7958 2581 46600 47466 61473 11929 79775 12581 89770 99455 83718 4628 16807 19335 81898 91552 86530 15811 79569 14148 3384 56954 8913 80114 16315 3686 89334 50382 1392 78326 71008 33553 70962 14957 14850 48231 22061 21185 71588 64305 2980 57564 41890 70052 68747 90943 81740 47593 87469 77370 12516 20443 96007 69525 3299 30412 57163 6699 63571 58976 87455 82217 93590 79343 14427 83220 41342 4584 84239 54496 65875 22601 31258 71449 43377 80040 37428 38238 72455 73517 71653 34433 11768 11957 27307 76456 26878 52977 45368 1999\n49935 850385413 998\n50039 997981495 999\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '125006655711\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
