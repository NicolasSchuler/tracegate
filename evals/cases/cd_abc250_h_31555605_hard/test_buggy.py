import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6 6 3\n1 4 1\n4 6 4\n2 5 2\n3 5 3\n5 6 5\n1 2 15\n3\n2 3 4\n2 3 5\n1 3 12\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\nYes\nYes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='36 41 22\n22 35 1\n13 22 1\n10 13 1\n27 35 1\n22 32 1\n2 13 1\n14 32 1\n13 30 1\n2 9 1\n27 28 1\n2 36 1\n30 31 1\n27 29 1\n7 32 1\n6 28 1\n21 35 1\n6 25 1\n16 22 1\n1 10 1\n7 23 1\n21 34 1\n6 17 1\n24 32 1\n19 30 1\n3 25 1\n13 33 1\n20 30 1\n7 12 1\n18 27 1\n15 25 1\n15 26 1\n8 13 1\n5 36 1\n4 6 1\n11 17 1\n14 26 1\n7 35 1\n21 27 1\n12 23 1\n18 23 1\n19 22 1\n84\n9 19 1\n3 12 1\n2 6 1\n3 12 1\n4 5 1\n2 10 1\n3 12 1\n9 21 1\n7 19 1\n2 13 1\n18 20 1\n7 10 1\n12 17 1\n6 7 1\n13 19 1\n3 20 1\n7 14 1\n14 19 1\n7 8 1\n1 18 1\n1 6 1\n19 21 1\n2 9 1\n3 12 1\n3 12 1\n16 19 1\n10 14 1\n9 12 1\n1 22 1\n1 8 1\n8 17 1\n2 3 1\n4 15 1\n6 22 1\n7 8 1\n8 12 1\n9 22 1\n10 11 1\n6 11 1\n5 7 1\n5 14 1\n10 19 1\n8 22 1\n13 21 1\n2 10 1\n17 18 1\n3 9 1\n13 14 1\n5 9 1\n10 17 1\n1 16 1\n4 6 1\n17 21 1\n11 16 1\n11 20 2\n11 15 2\n4 21 2\n14 19 2\n17 22 2\n17 21 2\n6 8 2\n15 19 2\n4 15 2\n11 12 2\n10 15 2\n8 20 2\n5 12 2\n3 10 2\n2 12 2\n9 11 2\n5 11 2\n12 19 2\n3 15 2\n13 18 2\n1 17 2\n14 18 2\n18 19 2\n6 10 2\n11 14 2\n1 11 2\n11 13 2\n5 10 2\n13 18 2\n7 8 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\nNo\nNo\nNo\nNo\nYes\nNo\nNo\nNo\nYes\nNo\nNo\nNo\nNo\nYes\nNo\nNo\nNo\nNo\nNo\nNo\nNo\nYes\nNo\nNo\nYes\nNo\nNo\nYes\nYes\nNo\nNo\nNo\nNo\nNo\nNo\nYes\nNo\nYes\nNo\nNo\nYes\nYes\nNo\nYes\nNo\nNo\nNo\nNo\nNo\nYes\nYes\nNo\nNo\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='19 111 15\n7 11 1\n3 7 1\n7 13 1\n13 18 1\n15 18 1\n5 11 1\n4 7 1\n3 14 1\n1 3 1\n7 9 1\n2 7 1\n2 17 1\n14 19 1\n3 8 1\n1 12 1\n1 10 1\n6 18 1\n14 16 1\n12 15 1\n7 17 1\n4 8 1\n8 15 1\n3 17 1\n8 9 1\n9 13 1\n1 6 1\n6 19 1\n10 15 1\n4 5 1\n1 7 1\n2 14 1\n4 17 1\n4 11 1\n5 10 1\n12 17 1\n7 14 1\n4 15 1\n1 8 1\n7 16 1\n15 16 1\n15 17 1\n7 10 1\n16 17 1\n7 8 1\n6 11 1\n2 6 1\n16 19 1\n5 8 1\n4 12 1\n1 4 1\n8 16 1\n6 7 1\n5 19 1\n8 19 1\n9 19 1\n8 18 1\n9 16 1\n2 12 1\n8 17 1\n11 15 1\n10 18 1\n7 15 1\n9 10 1\n14 17 1\n6 9 1\n7 18 1\n12 16 1\n5 6 1\n11 16 1\n5 18 1\n2 19 1\n3 10 1\n1 18 1\n6 13 1\n17 19 1\n3 11 1\n4 14 1\n5 17 1\n7 12 1\n7 19 1\n6 10 1\n2 5 1\n11 18 1\n11 14 1\n9 15 1\n13 15 1\n5 7 1\n10 16 1\n8 11 1\n10 13 1\n5 9 1\n10 19 1\n15 19 1\n3 6 1\n16 18 1\n2 11 1\n5 12 1\n3 18 1\n6 14 1\n14 18 1\n9 17 1\n3 13 1\n6 16 1\n9 12 1\n1 17 1\n12 19 1\n10 17 1\n11 13 1\n10 11 1\n2 13 1\n9 18 1\n71\n2 11 1\n1 3 1\n4 6 1\n3 9 1\n7 14 1\n2 10 1\n10 11 1\n4 6 1\n8 15 1\n11 12 1\n8 13 1\n2 9 1\n6 15 1\n5 6 1\n4 6 1\n9 15 1\n4 10 1\n2 14 1\n1 11 1\n12 15 1\n2 11 1\n3 5 1\n4 9 1\n9 12 1\n7 9 1\n1 5 1\n9 15 1\n4 14 1\n6 10 1\n8 10 1\n5 10 1\n3 15 1\n3 4 1\n4 7 1\n8 12 1\n9 11 1\n6 15 1\n9 11 1\n4 12 1\n3 15 1\n8 15 1\n4 8 1\n3 9 1\n11 14 1\n9 14 1\n1 14 1\n13 14 1\n9 14 1\n8 15 1\n6 13 1\n12 14 1\n11 15 1\n1 14 1\n5 15 1\n7 14 1\n13 15 1\n4 7 1\n9 15 1\n11 12 1\n5 9 1\n1 10 1\n6 14 1\n8 14 1\n2 14 1\n12 15 1\n1 9 1\n6 7 1\n7 10 1\n4 14 1\n2 15 1\n3 15 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nYes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='46 293 37\n24 33 62\n33 45 2\n39 45 96\n14 45 37\n18 33 38\n31 33 10\n19 39 99\n39 46 54\n2 46 77\n24 36 67\n22 45 58\n7 14 27\n34 39 78\n18 30 42\n18 42 48\n21 33 68\n40 45 97\n10 39 93\n6 45 23\n17 18 47\n35 45 18\n4 36 96\n36 37 7\n21 27 2\n4 13 84\n25 35 1\n40 41 30\n34 38 79\n11 24 87\n1 30 59\n8 27 32\n36 44 38\n14 29 12\n2 5 90\n10 15 87\n20 21 51\n19 26 84\n12 29 16\n43 46 84\n23 46 30\n3 31 73\n9 40 48\n16 25 7\n25 32 26\n23 28 62\n18 39 15\n30 33 78\n33 42 29\n12 27 77\n7 10 28\n16 32 89\n20 32 87\n7 34 78\n20 43 30\n23 36 41\n18 31 66\n29 34 4\n17 30 91\n16 36 78\n28 43 95\n32 39 37\n14 18 6\n5 39 34\n7 44 8\n41 44 78\n9 18 77\n15 41 87\n2 16 69\n1 2 70\n7 28 63\n29 39 89\n6 21 82\n38 45 52\n33 37 16\n7 8 61\n8 22 8\n4 38 8\n15 29 93\n42 43 5\n7 18 64\n2 13 59\n1 13 3\n2 29 79\n3 9 60\n13 27 16\n9 17 46\n7 41 70\n9 12 25\n16 34 36\n25 26 12\n3 30 26\n28 41 83\n30 45 44\n7 32 38\n10 29 95\n10 26 46\n20 39 87\n30 40 10\n19 37 79\n9 37 39\n2 37 50\n25 42 98\n8 42 11\n29 32 86\n12 46 50\n8 9 57\n30 39 89\n12 22 39\n12 39 11\n17 40 76\n13 33 44\n45 46 88\n16 27 37\n7 26 95\n17 34 78\n12 17 73\n16 22 27\n18 24 44\n29 43 69\n18 46 51\n1 4 78\n15 37 11\n3 41 33\n32 46 19\n5 32 64\n28 39 94\n5 24 15\n28 46 3\n27 32 35\n15 42 72\n35 43 6\n8 19 16\n3 11 4\n8 34 61\n10 19 26\n22 34 81\n10 30 94\n8 14 50\n10 32 67\n8 46 52\n10 12 18\n18 35 56\n18 29 77\n35 36 6\n9 15 61\n10 46 48\n2 21 64\n5 38 21\n12 25 89\n4 31 11\n27 36 14\n23 35 55\n5 46 72\n11 44 18\n43 44 38\n8 35 17\n19 23 47\n7 16 1\n28 44 9\n35 46 65\n1 34 67\n8 45 6\n5 45 15\n14 34 70\n16 31 14\n12 18 91\n41 46 99\n5 44 39\n25 37 13\n10 22 62\n10 40 9\n29 35 87\n30 31 7\n20 44 40\n26 45 40\n31 45 32\n5 19 29\n34 46 36\n15 27 20\n2 19 35\n2 32 44\n16 19 25\n13 23 61\n31 36 5\n10 43 84\n7 15 44\n9 25 4\n14 40 99\n12 13 15\n15 39 29\n2 45 72\n20 37 99\n25 40 96\n2 27 73\n33 44 84\n11 12 42\n19 43 26\n9 33 44\n10 11 19\n12 36 48\n9 21 23\n23 39 76\n12 43 35\n11 31 3\n14 37 19\n27 35 46\n23 33 15\n12 42 54\n23 29 40\n15 22 46\n7 9 66\n27 30 60\n5 18 89\n3 4 9\n8 38 98\n12 45 61\n10 18 57\n8 17 4\n3 25 63\n24 35 27\n18 36 79\n7 35 55\n37 41 97\n6 10 66\n35 38 97\n38 41 30\n39 40 53\n25 36 29\n18 34 80\n5 28 3\n14 38 1\n27 46 75\n3 20 66\n1 46 53\n13 16 72\n8 21 68\n25 31 26\n1 6 98\n33 41 23\n20 31 47\n9 28 88\n26 43 47\n30 44 84\n5 31 40\n20 29 53\n28 45 31\n3 12 87\n31 32 37\n12 20 29\n6 13 85\n44 46 44\n8 41 46\n13 46 87\n30 36 79\n12 41 89\n9 31 61\n6 39 39\n11 30 78\n19 25 85\n15 45 100\n31 40 72\n22 33 58\n2 15 22\n10 31 74\n17 39 90\n11 28 41\n5 13 35\n5 34 17\n7 21 5\n20 34 79\n15 31 97\n39 41 72\n36 41 46\n27 37 97\n23 24 73\n26 30 37\n10 33 23\n33 38 67\n21 36 68\n6 33 20\n21 23 45\n35 40 96\n24 26 39\n3 33 36\n1 22 53\n14 21 53\n6 34 2\n10 28 24\n9 39 83\n1 21 95\n27 44 26\n29 44 86\n24 46 28\n41\n8 17 3\n7 27 5\n7 21 5\n16 27 5\n21 25 6\n3 7 7\n27 30 7\n14 25 9\n17 37 9\n18 25 9\n11 17 10\n17 37 10\n30 34 11\n4 15 11\n15 35 11\n6 11 12\n17 26 12\n7 29 12\n22 29 12\n8 23 15\n13 30 15\n19 26 15\n13 18 15\n8 13 15\n4 12 15\n11 12 16\n18 19 16\n1 15 16\n11 19 16\n8 28 17\n10 34 17\n24 27 17\n5 11 17\n3 5 17\n10 23 17\n10 11 18\n27 32 21\n14 32 21\n1 20 28\n10 20 28\n20 29 29\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\nYes\nYes\nYes\nNo\nYes\nYes\nYes\nNo\nYes\nYes\nYes\nNo\nYes\nYes\nYes\nYes\nYes\nYes\nYes\nNo\nNo\nNo\nNo\nNo\nYes\nYes\nYes\nYes\nYes\nNo\nYes\nYes\nYes\nNo\nYes\nNo\nNo\nNo\nNo\nYes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='74 138 4\n6 30 72\n6 25 90\n6 29 18\n16 25 16\n16 48 39\n6 31 64\n6 19 63\n30 42 11\n16 49 40\n49 63 27\n27 29 37\n6 56 15\n26 42 41\n13 19 65\n29 72 74\n33 42 16\n6 43 2\n6 7 70\n4 13 53\n8 31 27\n19 44 65\n13 68 88\n55 63 32\n52 72 58\n49 60 98\n26 69 96\n8 36 96\n8 14 16\n52 61 38\n19 28 31\n28 71 30\n62 68 4\n18 72 15\n1 49 93\n36 57 7\n72 73 94\n12 18 29\n30 67 72\n3 52 20\n31 45 43\n4 46 26\n19 39 1\n2 16 94\n26 74 97\n49 58 19\n26 40 16\n16 20 30\n66 68 4\n16 53 67\n20 41 51\n9 39 57\n12 37 32\n47 49 33\n14 21 39\n35 42 80\n28 38 68\n60 64 50\n23 45 33\n66 70 79\n65 66 77\n59 66 6\n24 48 13\n15 66 29\n35 54 26\n22 36 11\n4 32 5\n11 52 54\n10 38 86\n12 34 97\n17 56 6\n5 48 54\n50 52 53\n32 51 57\n4 65 84\n9 24 44\n7 59 4\n43 71 63\n20 28 43\n17 29 26\n41 50 89\n13 38 39\n7 66 51\n30 70 76\n14 36 7\n14 63 32\n12 46 23\n31 38 12\n49 68 54\n47 71 74\n33 57 71\n32 33 9\n2 20 28\n2 21 27\n20 55 21\n8 42 3\n6 70 68\n11 59 95\n56 74 10\n50 68 85\n56 59 69\n2 42 58\n6 44 24\n2 33 6\n47 69 55\n4 41 89\n49 62 80\n14 26 77\n8 29 13\n30 34 77\n51 52 1\n38 39 86\n8 54 37\n58 65 55\n34 49 84\n36 67 54\n30 61 38\n34 57 61\n21 56 99\n50 61 21\n34 37 59\n35 41 41\n35 67 44\n64 67 34\n18 27 89\n7 62 47\n4 29 5\n17 67 64\n58 59 86\n24 42 36\n34 60 31\n20 66 14\n45 47 23\n47 61 65\n10 18 87\n33 59 74\n37 42 37\n29 36 11\n37 45 26\n7\n3 4 82\n2 3 82\n3 4 83\n1 2 175\n1 3 175\n1 3 175\n1 2 175\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\nNo\nYes\nYes\nYes\nYes\nYes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
