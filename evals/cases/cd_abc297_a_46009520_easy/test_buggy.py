import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 500\n300 900 1300 1700\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1300\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 99\n100 200 300 400 500\n',
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
        input='4 500\n100 600 1100 1600\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '600\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='74 31090202\n2133508 22455858 66583168 80995781 89181190 124030326 125821489 131817584 133370099 142216843 172906690 182235176 199694637 218783622 224316086 261870641 270751472 277835710 296391140 302565000 325318423 328744004 357562360 384285648 393430118 412974763 418366924 424380377 427227592 444801265 472082544 475403483 484106748 497364439 499540046 502842410 514914007 525165359 532225358 570427521 604331290 610245990 631431390 640316977 642805759 652068249 652161740 652337242 676930080 685596856 696014390 702001899 719905280 762426460 766868919 767804552 798793321 819626411 824565622 826797701 842193748 850358724 852063838 865200893 869226174 874351659 891340214 904583166 924233925 928008359 936503376 955320989 982797050 998416548\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '22455858\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='82 17836558\n2846555 7208269 15122889 47458029 49312684 59677948 59780069 80644148 80663916 88938518 104169306 117856659 132400484 137409143 141045311 155863663 164129927 208203454 208353248 212661331 222583794 291029596 293358618 320944299 324005776 330434068 338008567 352819193 367511979 370905807 380178679 440413930 445858410 461867893 481043790 494175118 524965338 534027166 552243811 557091916 559636285 561946825 574353637 575544659 594039530 597401248 610853878 620281860 629827384 631615920 643414188 649063641 652031280 652658581 656246552 669420947 700789399 705653853 734523721 742612832 744276525 767316674 772395003 775006307 786193106 800329109 815739254 817682109 836936876 890292851 890545876 899199725 901478538 902401555 907253554 910660366 912737725 929581038 943334070 960043815 971852487 992644480\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '7208269\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
