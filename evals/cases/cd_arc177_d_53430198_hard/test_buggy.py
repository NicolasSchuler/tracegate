import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='189 1\n1 65 129 3 67 131 5 69 133 7 71 135 9 73 137 11 75 139 13 77 141 15 79 143 17 81 145 19 83 147 21 85 149 23 87 151 25 89 153 27 91 155 29 93 157 31 95 159 33 97 161 35 99 163 37 101 165 39 103 167 41 105 169 43 107 171 45 109 173 47 111 175 49 113 177 51 115 179 53 117 181 55 119 183 57 121 185 59 123 187 61 125 189 60 124 188 14 78 142 12 76 140 10 74 138 6 70 134 4 68 132 2 66 130 62 126 190 58 122 186 56 120 184 54 118 182 52 116 180 50 114 178 48 112 176 46 110 174 44 108 172 42 106 170 40 104 168 38 102 166 36 100 164 34 98 162 32 96 160 30 94 158 28 92 156 26 90 154 24 88 152 22 86 150 20 84 148 18 82 146 16 80 144 8 72 136 0 64 128\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 679058953 749268343 373873218 119073186 238146372 217190597 162751213 325502426 107744890 620362835 484962634 700894930 743373419 488502485 658416441 438944294 877888588 0 0 0 0 679058953 359873553 81376306 244128918 488257836 141023931 329055839 658111678 411994957 455027328 910054656 83939787 572773301 147302249 106147599 602136483 206028613 193495443 485133133 970266266 140328320 745512901 492781449 917031067 843328665 688412977 393603369 108155013 216310026 72760955 4101900 8203800 731811933 58051488 116102976 415794255 517224793 36205233 441927385 259807962 519615924 789382585 349394058 698788116 935322714 485037573 970075146 167178935 857435283 716626213 482918546 215499115 430998230 359564859 18437376 36874752 668853526 141523757 283047514 170217144 765361960 532479567 916570125 974996053 951747753 187207112 810326843 986297252 785851908 438944294 35398233 70796466 141592932\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 2\n0 3 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4 2 2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 10\n10 55 20 45\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 4 4 8\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='8 1\n5 0 6 3 8 1 7 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 64 32 48 24 28 28 32\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='40 20\n695 793 11 502 114 861 559 4 212 609 894 435 429 94 91 258 161 45 33 605 673 634 629 163 283 826 409 84 507 548 31 248 588 340 357 168 926 949 322 912\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 41942627 41942627 360709869\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
