import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n2 1 3\n3\n2 3\n1 2\n1 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n2\n1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7\n3 7 2 1 6 5 4\n13\n2 3\n1 2\n1 3\n3 6\n3 7\n2 4\n3 7\n1 3\n4 7\n1 6\n2 4\n1 3\n1 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\n2\n3\n3\n3\n2\n3\n3\n7\n1\n2\n3\n3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n1 2\n1\n1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n94 32 91 9 39 31 59 67 66 22 1 93 40 98 86 11 89 61 79 96 5 19 90 30 88 56 68 74 27 20 42 72 44 15 87 75 14 29 92 81 62 76 46 18 51 12 84 21 53 24 55 83 37 25 3 85 69 17 36 26 50 43 33 57 80 7 58 60 64 35 99 54 71 2 34 23 4 38 77 73 65 95 97 49 48 13 45 52 6 78 41 70 100 10 82 47 8 16 63 28\n100\n34 63\n2 36\n49 51\n65 90\n42 45\n33 71\n43 98\n48 56\n39 49\n3 97\n12 65\n30 95\n39 86\n85 94\n33 56\n83 98\n53 77\n27 35\n64 95\n37 90\n56 70\n18 19\n5 16\n4 61\n77 97\n63 74\n41 59\n38 44\n38 76\n9 16\n11 23\n11 30\n33 60\n70 85\n18 43\n2 62\n13 30\n27 73\n60 72\n75 84\n49 54\n16 83\n73 91\n84 88\n36 49\n1 53\n66 80\n20 86\n9 72\n21 92\n12 72\n41 89\n11 87\n35 69\n43 88\n78 84\n24 86\n52 93\n28 78\n10 27\n63 77\n22 75\n24 84\n20 88\n77 82\n83 92\n55 73\n31 79\n43 66\n2 9\n7 61\n33 100\n20 60\n17 92\n19 72\n66 91\n75 90\n52 94\n1 87\n55 59\n60 96\n56 61\n23 54\n23 26\n9 49\n29 86\n6 70\n65 72\n1 72\n77 95\n30 52\n33 82\n14 46\n43 92\n50 81\n35 71\n10 66\n43 88\n38 81\n30 33\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '34\n36\n51\n90\n42\n33\n98\n56\n39\n3\n12\n30\n39\n94\n56\n98\n53\n27\n64\n90\n56\n19\n5\n61\n77\n74\n59\n44\n76\n9\n11\n11\n33\n85\n18\n62\n30\n27\n72\n75\n54\n83\n91\n88\n36\n1\n66\n86\n9\n92\n72\n89\n11\n69\n88\n84\n86\n93\n78\n27\n77\n22\n84\n88\n77\n92\n55\n31\n66\n9\n61\n33\n20\n92\n19\n91\n90\n94\n1\n59\n96\n61\n54\n26\n9\n86\n6\n72\n1\n77\n30\n33\n14\n92\n81\n35\n66\n88\n81\n30\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n35 47 50 83 27 60 56 18 49 12 58 28 20 54 98 77 48 95 52 69 71 82 55 85 61 17 38 41 15 57 59 42 99 86 34 76 53 13 67 10 70 97 90 74 96 26 64 75 4 40 65 66 7 62 87 5 14 81 44 39 3 1 32 94 11 37 92 24 33 88 29 91 72 45 63 93 68 9 21 73 51 79 8 46 100 36 16 89 25 80 43 6 78 19 84 2 23 31 30 22\n100\n46 85\n53 91\n5 10\n53 56\n43 94\n54 97\n31 75\n20 74\n37 95\n45 54\n6 73\n24 65\n31 52\n29 53\n31 92\n13 27\n38 81\n58 63\n58 66\n1 95\n7 74\n51 68\n9 78\n4 7\n8 92\n50 99\n6 99\n22 98\n15 81\n67 82\n19 37\n11 26\n44 88\n26 44\n2 88\n44 55\n21 89\n31 46\n16 45\n34 54\n22 50\n42 83\n26 41\n1 86\n32 61\n61 68\n3 76\n58 72\n59 66\n24 36\n41 54\n18 25\n49 95\n25 56\n50 78\n11 83\n84 90\n45 59\n45 68\n1 22\n55 72\n75 85\n12 65\n20 21\n16 96\n49 57\n7 48\n35 91\n82 84\n68 98\n41 74\n39 93\n3 17\n34 68\n65 71\n21 83\n3 66\n32 49\n76 88\n38 64\n36 73\n22 70\n32 74\n55 60\n48 80\n61 89\n20 54\n33 34\n45 90\n12 40\n17 35\n11 37\n44 83\n68 78\n45 56\n27 98\n8 28\n58 66\n71 91\n5 76\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '85\n53\n10\n56\n94\n54\n75\n20\n95\n54\n73\n65\n52\n53\n92\n27\n38\n58\n58\n95\n74\n68\n9\n4\n92\n50\n99\n98\n15\n82\n37\n26\n44\n26\n88\n55\n21\n46\n45\n54\n50\n83\n41\n86\n61\n61\n76\n58\n59\n24\n54\n18\n49\n56\n50\n83\n90\n59\n45\n1\n55\n85\n12\n20\n96\n49\n48\n35\n82\n98\n41\n39\n17\n34\n71\n83\n66\n49\n76\n38\n73\n70\n74\n60\n48\n61\n20\n34\n90\n12\n35\n11\n83\n68\n56\n27\n28\n58\n71\n76\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
