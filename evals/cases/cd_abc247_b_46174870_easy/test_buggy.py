import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\ntanaka taro\ntanaka jiro\nsuzuki hanako\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\naaa bbb\nxxx aaa\nbbb yyy\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\ntanaka taro\ntanaka taro\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\ntakahashi chokudai\naoki kensho\nsnu ke\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\nfjdqeixfum cwxfmekhcl\nagitvtophb kilakuupco\nepfepxbfgs nqxeqvfmma\nqcugugpugt puvccorwui\nsxzqbvfhxy kotlkhfqar\nehfkuqyfwt shhyrtsaaq\nmglrjaiten shsoyksuhw\nfqiqacwvuf yatebtovzp\npfvvjhaddg xecwdcvnrx\nlpvaaacxsy xkndbbgbch\nqvojuiosfy eensshrtnz\nsuzcakihtl sdjrdhauic\nslinukpxzi bxfkgidazo\nueboqoennj iudypxskal\ndhvixbgcgf zsyputhkgs\ndkfuxjxldn evqpaoffni\nxnwlpdljkt hszhahvqxf\ncsfyojkdkk mdyprzmvcz\nnwtaiweonq pfrqknklgd\nqajxwsmrnk cvetawzidf\njhkwevmncn lxqtyclyrx\nwjmooyonrl jhimyyltco\nultnjuijft fumuftoaqx\nhbpxnwnamm esyrawomeq\nswzyrdlxbf owxbgtikfr\nrqzpxhublf wxwkdqmpzv\ngdgxtfozsg qptcnldhkr\nnwieibnqey hrhidokjtc\naxefdlqrra hlcvvukxhk\ngkbxjepagp wofiiujpdv\nskqzfritwz vsubmrrymu\nuzagqomwlf tzyamoqunb\ntkjkgsgdry xlmizdkqab\nwwlcocllno mzlftnitcd\ndpqeuupmos qjbkfazgdj\ngjwbidcidg nlpiipmlou\nfnezvqoiro lknjvsdrzi\nmvgmwbrajo lgoiaydugh\nzryysfzgxx zeetmuizni\nycldlsuwci dkcgjzlokq\nznslmdwcfu pyiofsqspv\nriuhuclufd yuqwxgvbmh\nlcspmjnpcq eedchinlrg\ntkvdczzyqe pxydmpeojs\nnqdkjexxtw fxtgqaitct\nypkejgjtgr nwakxdwhdi\nsuyimnfxez exufqvnqtu\nedccogcshb gtehyqwqlv\niyrtxyjbsg ouvwxsnzoy\nzmkpxkpsgp sutmopmekb\nwrorateswi cmhfgkmfks\nxhrzngxtay dnknmohgqx\nujvehmokce knrmyxcydc\nfgkncgoage zwgmscnhru\npeqshnovgd abcnlogboe\ntitavtlcom dhzlqorpzm\nxpxenbvyoh msqrldsjot\ndlqmqtvvlc xbnnazlery\njbvfjzhnwy cflqyadlte\ngsninllcxx ynekzraptw\nkofdosbtvt klekeowouv\nfrrduuoaan kromosdgxr\ndhmgfhqbtx aybtgnbazb\nhpxsupawtf sllubioypc\nokwmqoyzhr zbwolpahhe\nsslbmarmrd bmultxhzqy\nqajwrzsvqy qrqrxrloro\nwiqlrnlvqv jtfvuhpope\njjqupnwabh zbwdyyapby\nzddrgnnpmy dofoiylddd\nivvgmexsvw hantmcwsom\nzhiqoqxrkj vxnxwpjvoq\niexbjxqnwk ostywrwial\npxtpyppwgh qtrnavffub\nvvzgwefalv lzrhfvdwvi\ngwsphzryde nvothlwnyu\ncfxyvzckcv bpzuigulrc\nzykejgxcjl rvzlnsqyoz\nanihskfgxl hntuippilg\nqyutnqqglr ogguwxmgcg\njdyenzhefk ekwgdrbxzn\ncqmcgbqgwe cvzrunnvke\nnlsckhioxu blfiukfnwe\njfylbvaypu okneginobj\nxpdlxqgayz emekisinlc\nbjpqlsafor bkyjslfxvr\nclawfrcrub vzbwvszdel\nvqydylsugs paotbyqfac\nzcikfzlyhs jsshmkouos\nzryepifqow umbrvamqsu\ntljwgxgtjx ozavbidblg\ntzjyfjhxff jxyyniejqo\nuksvrawosq setzrnnqoa\npevwpgjzim qvgitruige\nbrsfkvpehh nteigwjmke\ndobxzepudg nedyktsewy\nemeiqfjppg jvfytgtetb\nspaiwwasox ziezgffxda\nspbujdztde xfiwechcre\nsluhdjllvk tpeepksfpe\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
