import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1\n1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10000 1\n1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 3\n1 3 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '6\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10000 100\n1 63 84 102 238 399 481 525 536 559 609 717 751 803 818 921 956 981 1014 1015 1018 1032 1037 1186 1211 1327 1350 1351 1441 1504 1509 1515 1570 1754 1856 1880 1931 1949 1961 1997 2079 2234 2245 2262 2328 2434 2452 2484 2494 2500 2611 2617 2643 2773 2796 2804 3018 3050 3061 3165 3199 3204 3268 3324 3366 3610 3636 3640 3721 3755 3806 3811 3823 3824 3836 3942 3951 3982 4032 4039 4061 4090 4103 4174 4188 4297 4345 4352 4362 4460 4501 4589 4610 4626 4708 4864 4882 4890 4900 4929\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5071\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6104 100\n1 18 35 47 49 57 60 62 63 65 80 87 106 109 116 127 163 168 180 182 191 192 193 201 205 207 210 211 218 220 221 226 236 242 251 255 260 262 264 270 271 276 277 283 285 297 300 305 317 326 329 335 344 346 351 353 360 368 369 371 372 376 378 385 390 399 408 418 424 438 443 448 458 459 460 466 473 479 482 483 484 486 487 489 490 493 495 496 498 500 505 509 512 525 527 532 540 550 574 590\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3154\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
