import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 2\nred green blue\nblue red\n800 1600 2800\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5200\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 2\ncode queen atcoder\nking queen\n10 1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '21\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1\na\nb\n1 1000\n',
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
        input='22 75\nsjrggtycgcqfpo mrouhjkntqryhji x whfut mjcdlivnuxmrdf vwulbdkc ncksizgzajkhyjujlks yluacvrayggw ekfdsqcfvmqbocbomj lcgs tft l ks ycebeurgvjyospdhvuyf ubspxsiuytqrwmv yfezuja arenexc ueahebyvsak agmaygzyfnzpqftobt gpuamk yluacvrayggw mopodotkrjzri\nagmaygzyfnzpqftobt ak amahn arenexc bbo cdqciuehvfndghrxxnm cyurw dcc dpjimsfbuslks ehxmoy ekfdsqcfvmqbocbomj eycanjlbe fazdprco fbagzrxkprtgbo fdrsqfseqsdfcgdenmrf fnd fsvzlotlfw gpuamk hjphvs hk iblsrjuyfskndapwjef jdcqcrxvyaherm jtepxhnve kfxiwhfzfksnawg ks l lbdnfhfxamo lcgs ll mliygnfddeswmxuvh mopodotkrjzri ncksizgzajkhyjujlks nviaztpcmxlvova ofklitxaoykfoqk otuayxmwv ozjaraefml pdzfjmcjcmsaaskex pnzijdifyervjaoalc ppgykichjt pphykinkezplvfja qjhwkcexecmdk qqtqbxjjmqn qrjolv qv rahdigyoj rnhrhctntzly rqsfskevfxphcqoq s ssjoatryxmbwxbwex szafzjmdtsbyldhylcg sznjyarkuprerivhub tft tgqngguftuahalwvjw tvnr ubelzv ubspxsiuytqrwmv ueahebyvsak ujidhukzwzc ukfnlxfcevkjezqs um veergcgmonctcsvypol vwulbdkc wkrpiuiasfkjexpftzn woeyom wseqzd xovldqfza ycebeurgvjyospdhvuyf yf yfezuja yfkilrixodiwqcyre yluacvrayggw yngvmhneqyjdu zakbzrkjwqdyuxdvo zg zljxioyundvpndabsb\n3889 2778 7931 6678 5919 8952 10000 49 3547 1261 7434 951 7318 8430 7391 3993 7823 4503 1963 2346 4972 2406 3854 661 3031 989 5851 8228 5263 1412 7499 9178 819 2436 3672 571 4795 5549 1554 3967 9215 289 2925 6019 6141 2583 3969 119 5015 281 1993 7200 8835 518 3781 6616 5865 9920 8504 3545 7881 783 8490 214 2082 2285 1943 307 1164 9976 3933 3047 6095 3812 6978 8593\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '102643\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='14 22\nf nqwid zqiygwf efiwpx umdwugbfgno b pafgkirrueoezabcql cdwmklvyzvuughetw dnwhrssyiybitcoowlhm lbzlnvmosmvqub nrqa hacqfjcmwupoxonavvx rdpgtrdwalvffknwibh svprpymskznab\naqhsxzguseuexlycgub b bqgcrgvwalcptutzxs bxrmqdgqjsosencrbwk ddflgtixbnagkqrzes dnwhrssyiybitcoowlhm hacqfjcmwupoxonavvx krihwqplf lbzlnvmosmvqub lkchithcdezsaef llvucvubyozfmttxtlsr ltvmmzmslqdxqiifz npjgqftvwgukjwsekfcg nrqa quwssucxrmywerf rdpgtrdwalvffknwibh rufxdsqchjkuxpxng shaszrenhlbz spkq svprpymskznab ws ziqdziycnynthgmdav\n823 2805 732 9390 8151 8826 10000 4576 7174 4485 5046 7157 4375 4223 6693 1696 1083 8099 9938 8968 2778 8723 4141\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '36108\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
