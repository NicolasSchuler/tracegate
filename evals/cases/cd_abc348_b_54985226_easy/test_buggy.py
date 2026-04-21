import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n0 0\n2 4\n5 0\n3 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\n3\n1\n1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6\n3 2\n1 6\n4 5\n1 3\n5 5\n9 8\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '6\n6\n6\n6\n6\n4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='33\n621 942\n-911 438\n-424 -344\n-384 733\n503 837\n-520 120\n950 36\n397 28\n-51 213\n144 -384\n141 -363\n-659 163\n281 -193\n972 -830\n-582 -192\n-496 -171\n533 67\n-987 389\n-128 -960\n-505 51\n458 286\n213 220\n112 -919\n715 430\n-931 -188\n413 945\n-155 329\n810 -626\n40 -828\n434 398\n-537 -281\n654 -177\n-947 -715\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '33\n14\n1\n14\n33\n14\n33\n33\n14\n1\n1\n14\n18\n18\n14\n14\n33\n14\n1\n14\n33\n33\n1\n33\n14\n33\n14\n18\n1\n33\n1\n18\n1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='57\n-541 -828\n-542 -956\n-743 -391\n-450 -662\n-628 -764\n-945 -650\n-62 253\n-460 754\n-576 247\n-872 -736\n-928 678\n739 20\n411 127\n869 629\n332 233\n823 218\n-756 -495\n312 -23\n153 -341\n496 697\n316 -630\n423 -501\n971 -960\n327 993\n664 -639\n871 -414\n-863 -266\n689 992\n152 350\n355 -58\n575 -684\n680 -601\n68 371\n251 -371\n-139 -956\n297 764\n-694 862\n167 -223\n-105 557\n-837 100\n195 -917\n-174 14\n533 -789\n-745 -480\n774 787\n368 444\n437 790\n681 831\n-94 -805\n-683 -6\n-786 367\n-379 970\n503 551\n-39 342\n-847 744\n-771 70\n-320 871\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '28\n28\n28\n28\n28\n28\n23\n23\n23\n28\n23\n6\n6\n6\n6\n6\n28\n11\n11\n10\n11\n11\n11\n2\n11\n11\n28\n10\n23\n11\n11\n11\n23\n11\n28\n2\n23\n11\n23\n23\n37\n23\n11\n28\n10\n10\n10\n10\n28\n23\n23\n23\n10\n23\n23\n23\n23\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='66\n-375 -659\n-1 -816\n380 -211\n-50 772\n-51 619\n181 -440\n428 -951\n845 -16\n-739 899\n335 -390\n-507 347\n713 311\n294 292\n-643 972\n-73 538\n267 565\n388 547\n-741 214\n-305 544\n-265 -279\n823 542\n-636 566\n-81 330\n844 -522\n294 178\n686 159\n47 -901\n-6 -346\n345 -97\n-315 418\n819 47\n-400 652\n679 -381\n637 -684\n310 467\n-651 731\n-141 -453\n-468 -265\n-960 714\n-908 -768\n-440 -770\n590 71\n-25 -216\n-102 204\n-430 55\n595 935\n-467 283\n963 987\n491 -163\n-73 909\n481 -550\n-118 -337\n379 830\n89 607\n546 218\n262 461\n787 626\n390 -12\n578 299\n-738 -923\n-719 84\n817 -880\n454 -541\n-487 -971\n609 -679\n-881 24\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '48\n48\n39\n62\n62\n14\n14\n39\n62\n39\n62\n40\n40\n62\n62\n60\n60\n62\n62\n48\n40\n62\n62\n39\n40\n40\n48\n48\n39\n62\n40\n62\n39\n39\n60\n62\n48\n48\n62\n48\n48\n40\n48\n62\n48\n60\n62\n40\n39\n62\n39\n48\n60\n60\n40\n60\n40\n39\n40\n48\n48\n39\n39\n48\n39\n48\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
