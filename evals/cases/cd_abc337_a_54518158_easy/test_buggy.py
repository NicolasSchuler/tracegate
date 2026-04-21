import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n10 2\n10 1\n10 2\n3 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Takahashi\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6\n5 4\n4 5\n2 4\n1 6\n7 1\n3 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Draw\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n0 0\n10 10\n50 50\n0 100\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Aoki\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n3 83\n59 13\n73 64\n69 2\n62 26\n64 59\n58 89\n78 68\n30 37\n80 47\n36 98\n57 44\n97 73\n71 53\n7 79\n57 29\n81 41\n89 36\n35 55\n31 52\n21 35\n91 15\n49 42\n57 100\n56 12\n74 81\n15 96\n40 94\n32 23\n48 65\n26 36\n66 20\n49 86\n94 82\n14 29\n46 77\n74 67\n89 20\n12 1\n75 50\n57 95\n27 30\n33 47\n29 24\n43 91\n80 41\n9 55\n37 96\n56 38\n12 53\n47 8\n3 0\n61 10\n100 51\n17 10\n34 85\n79 27\n48 9\n43 60\n72 64\n18 73\n76 56\n83 85\n98 91\n100 24\n97 45\n56 26\n45 85\n20 62\n19 78\n17 48\n56 24\n82 67\n43 37\n76 36\n53 29\n61 46\n15 59\n82 24\n7 41\n1 95\n41 45\n36 5\n30 33\n9 23\n4 38\n99 15\n11 39\n24 81\n37 44\n69 80\n90 26\n13 9\n37 13\n76 90\n56 99\n48 70\n17 77\n29 54\n78 16\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Draw\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n9 98\n98 70\n17 34\n9 25\n47 14\n50 32\n79 15\n52 44\n41 56\n80 33\n73 5\n35 43\n95 75\n18 34\n76 21\n49 80\n14 99\n23 62\n5 66\n34 68\n43 36\n30 15\n92 96\n89 47\n66 94\n70 86\n37 20\n71 10\n97 14\n12 78\n78 68\n8 39\n58 17\n67 65\n36 77\n60 70\n75 11\n51 51\n18 21\n83 44\n30 84\n51 1\n96 19\n27 13\n12 21\n11 41\n58 62\n40 31\n85 39\n28 23\n84 19\n49 90\n40 19\n92 8\n50 93\n2 62\n19 66\n43 96\n30 73\n85 10\n33 7\n86 30\n15 94\n27 39\n68 80\n98 73\n56 30\n13 37\n33 96\n30 1\n16 97\n81 99\n7 22\n12 77\n44 100\n4 96\n12 79\n92 82\n58 86\n85 55\n63 31\n82 77\n73 86\n9 18\n77 10\n86 59\n32 48\n96 91\n25 46\n14 5\n3 55\n15 26\n9 17\n60 55\n89 54\n47 52\n71 7\n54 40\n95 56\n24 73\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Aoki\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
