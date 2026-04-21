import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\nalice 31\nbob 41\ncarol 5\ndave 92\nellen 65\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'carol\ndave\nellen\nalice\nbob\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\ntakahashi 1000000000\naoki 999999999\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'aoki\ntakahashi\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\ngmsrreobv 0\nhcjqy 1000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'gmsrreobv\nhcjqy\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\nd 1000000000\ngnuhyzlmr 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'gnuhyzlmr\nd\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\nmsre 0\nm 465342399\ne 934412340\nla 549477183\nf 309293399\nltjbtevtko 553180947\ncgcm 983769960\nrvmmwumexs 920319986\npn 238378982\nvqqg 536015685\ny 678354458\nfynwq 695999238\numadbqvboc 664690177\nyywknv 124476960\newgwkr 497817741\nftqpssbihu 98815987\nbkp 809759746\nuc 679202905\nvju 788724124\nvnc 926257684\nzhjxpkuqz 911003576\ngxe 260018272\nppwfdvuzp 174150682\nocaopiswx 892418243\nrtxkgsauy 663620539\nrvnpit 991742938\nogjk 717234670\nwkhwyeegig 346984751\nukehwghlii 326250733\nq 566456784\nn 165381265\nisbn 827545211\nmpxqectouy 346487189\nwlfmhhx 969284206\nfautyzmus 83356755\nwkzzmpu 924468534\nusumayoi 467103859\nrvfchnoh 40588379\nc 899481137\nsg 922068726\nsd 842390249\nshrmhql 955648488\nujjnedxz 427062584\nsntiddvcxu 278417579\nwdbu 296155209\nsifhkqf 681102205\nhzizzq 593459358\nkmodlt 829010878\nzxis 763411540\nhwgcrql 691224423\na 861463413\ns 984992425\nignvndzn 640816993\nkhw 275683577\nopjgphh 106777755\nzexvo 271791780\nwetf 49557947\nfvfrswe 492590166\nqornridtz 364178413\nbvut 179043109\njuaxa 340395518\nkcd 400539269\nkduxu 929017310\niuhiygatob 70060532\nbklsqfhalq 57062870\njcdc 557351347\nswx 218938542\naahohhpzf 996468862\nwvbrvb 897914734\nzs 485100103\nhdyhwk 647696503\nmarkfdakw 815798173\nbmhqwcmm 733100712\nxok 454017831\nkzfprefgpf 62297160\nweoqvz 560182600\nkwyx 403150469\ncbdno 725282543\ncjohnjecx 508552113\nhbfixfz 853824818\njahfith 857715490\nb 850392958\nx 498040132\nbtxrsnrxd 94140502\nkbj 17002776\nwww 548517076\nza 747214124\nwezjrf 777126853\nsvmbhpotoa 561912442\nfcn 876535343\nmbmqs 856539552\nu 158703675\nkmwrnsyls 474311824\ncqhnpjw 445803289\njff 615432286\nkoiscwin 457770243\ngif 564440520\nhlcmrypyyt 252871692\nnhyqsj 497880638\nnm 265231775\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'msre\nm\ne\nla\nf\nltjbtevtko\ncgcm\nrvmmwumexs\npn\nvqqg\ny\nfynwq\numadbqvboc\nyywknv\newgwkr\nftqpssbihu\nbkp\nuc\nvju\nvnc\nzhjxpkuqz\ngxe\nppwfdvuzp\nocaopiswx\nrtxkgsauy\nrvnpit\nogjk\nwkhwyeegig\nukehwghlii\nq\nn\nisbn\nmpxqectouy\nwlfmhhx\nfautyzmus\nwkzzmpu\nusumayoi\nrvfchnoh\nc\nsg\nsd\nshrmhql\nujjnedxz\nsntiddvcxu\nwdbu\nsifhkqf\nhzizzq\nkmodlt\nzxis\nhwgcrql\na\ns\nignvndzn\nkhw\nopjgphh\nzexvo\nwetf\nfvfrswe\nqornridtz\nbvut\njuaxa\nkcd\nkduxu\niuhiygatob\nbklsqfhalq\njcdc\nswx\naahohhpzf\nwvbrvb\nzs\nhdyhwk\nmarkfdakw\nbmhqwcmm\nxok\nkzfprefgpf\nweoqvz\nkwyx\ncbdno\ncjohnjecx\nhbfixfz\njahfith\nb\nx\nbtxrsnrxd\nkbj\nwww\nza\nwezjrf\nsvmbhpotoa\nfcn\nmbmqs\nu\nkmwrnsyls\ncqhnpjw\njff\nkoiscwin\ngif\nhlcmrypyyt\nnhyqsj\nnm\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
