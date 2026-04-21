import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\ntakahashi 2\naoki 6\nsnuke 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'snuke\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\ntakahashi 2813\ntakahashixx 1086\ntakahashix 4229\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'takahashix\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\ndlshogi 297\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'dlshogi\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\nkxaasyl 821\nncgrlupxn 415\nmevehydqsh 2550\nwnzomxmncwcawo 1586\nqoucugsrintl 2567\nqjytjrweyvyq 1958\nijqrrknlsfdjdk 4135\nysylpnyxhet 3714\neatmjkfeovysmujj 2822\nobl 4213\nisqyrlwwobs 1441\ndxrb 3819\ntdqatz 1718\nclph 3557\ndvo 3095\nzejauldhgfpro 1691\nggyg 1069\nahbr 1861\niimjcky 1764\neephkixrdbixj 1410\njhjjhmhkev 789\nbgjcl 1327\nibpjmdxtld 756\nhalbnibdbzkfwroc 2056\nsosgkyoma 1245\nzwqelcbrjcw 536\nnjpaicybb 105\njswshrhcskeyojkq 1509\naqhy 2821\nxaxafvpqivcmuf 1248\npcmviqwujnljun 1428\nyrcoqtfptcyrsi 1833\ncyzevthx 2187\nhtxffcbcrujubedb 1841\nzloswvyn 1974\nhgndrpk 1644\nxlsjbbspfj 1898\nqsebsconh 2247\nhwjuqaxz 4178\nrmjkkg 3741\nvyhhwqlzdzgt 2040\ngfgic 1270\nplygvmbuij 1686\nygc 1274\nfxggyenndha 2253\nqkmm 1805\ngplwvrtvbv 3271\nfixeiwfnwkktaqor 3108\ncamna 2941\nziwoqtzotzrqeez 487\nsomcsm 2401\nkzqzj 3037\niirytsclhhznv 793\nddikadcr 269\nvdjsxfihg 1546\ndaiinsce 1229\nhhltcvlctl 1420\naoushbcgitcdwwqz 3067\ntsl 1431\nssy 3541\nzcxigfcbnqbjxl 847\nqiutrj 1769\ndemexwfc 1629\nvsxfhatb 2146\nhio 3765\nocrksiaj 3162\nddplkkwspvg 2349\nhmag 1248\nvcgf 334\ntxyws 3246\njzlaqfyjdvvxp 2557\nuuwftvjtnha 3016\nojxa 2297\ntivoorlnjcj 732\naymsod 586\nhzfkafxjkrvs 2049\nwnpznduvpxpe 2037\nfevtpitltjx 2279\nrhhmam 1853\nqneqd 317\nqmgbbpwelkwjjk 207\nvxsybdhlayftfiq 3008\nnxc 1576\ngkahpll 3213\nhdgrecamrpmu 2474\nilmcumyycvw 14\nfiqllxzoowsty 3911\nkwy 1984\nqfquafnwr 1888\novhdg 3310\nguatmh 1564\nxwoyppzf 2808\nsgfpyauzlrihnp 2564\nmzajrgevdosyk 2261\nntdw 1985\nntcmbgyhdpua 1893\ntvvnyuyjpccar 1797\nrcpvsolxqpjz 1403\naje 1169\nhmztkptzsgtmvb 15\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'ahbr\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='41\nkkdoflhmltubpsdk 2916\nzunxvceded 2612\nsisapeagdva 4142\nseufrtkodesn 3424\nnvryfanysubfmt 2735\nzobxjfvrzmalmid 1466\nqkqeuz 3298\nfvrplet 2709\nhopkolazcp 962\notxuhbbttrsaw 3078\nnydkkqc 44\nmvbf 2272\nhcz 54\nazk 3245\novwchjwtu 2999\nznddo 438\nfcvessj 1816\npjflxhpxczy 3812\nzxmrsklmry 396\nudu 1586\njbxdrkancetkg 1051\nvckakcnfhtcpg 1750\nohmxvdegbk 1223\nzcv 871\nrwslmstt 3751\nixzqm 1523\nawsrcmxbxzslsgd 4193\nwriobzgqx 367\nfvvrwyjgkag 1035\nsloqwhlxueuy 1650\ngqqphyg 1178\nqsgi 1740\nypvva 3704\nwwifsvxstbcqkikb 998\nfufafvlmpmijfbem 1621\ndjfvymcwu 3060\ncnddugugrkqtgvp 2114\nehf 468\npheahhgggp 3303\nndmun 83\nuwsbmgqin 894\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'ndmun\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
