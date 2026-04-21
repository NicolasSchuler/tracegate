import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\nab\nccef\nda\na\nfe\n',
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
        input='3\na\nb\naba\n',
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
        input='2\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\ncb\nabcd\na\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\nlkqcepsnljbyoyxchdiyygwchtcoxyzzblncvosgcffrhasevy\nqcyreiwkrpiuiasfkjexpftznqiblsrjuyfskndapwjefucdqc\nnjwjcrqayjpcijndyzmctzbaoqeugwrmhpkzvyygkovpckbzet\nqlnxmetghtnrefeypivpyndobbresgvlmkkxwhlomigirizlng\nxrqsfskevfxphcqoqkbbomcyurwlrnhrhctntzlylvwulbdkcd\nrenexcfycebeurgvjyospdhvuyfvtvnrdyluacvrayggwnpnzi\niuehvfndghrxxnmvzljxioyundvpndabsbnwoeyomrjdcqcrxv\nhmzponrtkqjuanpnufnluezljsqvtofmenwfzlgrrpzetxogob\nfiwpxueumdwugbfgnojjrjpafgkirrueoezabcqlzmcdwmklvy\nwhrssyiybitcoowlhmmrdpgtrdwalvffknwibhwhacqfjcmwup\noqfqbweuchtdvpiunkpehcshtmrentgsndnbtbbbmoguopypwk\njceofctzqygfietzkbegkbsjzmfzrmfpmgvoafxfyinmayuxcj\ncgdenmrfwfndijtepxhnvedfbagzrxkprtgboukfxiwhfzfksn\nbztsxaeunetdtikjbbxuqjcdwyhwregfyjhctuarlqofgwhpyu\nnoixwparklayjdsbijtrgtxgzfcpuchdsmkvqrhgdiidnnunws\nxdvoossjoatryxmbwxbwexnagmaygzyfnzpqftobtaotuayxmw\nskctdjfquvekmckrwivzayctxcfxptgruprmpnzswuoegwgdbb\nitmdfbwzhiowcpggbunwiopnjdjcwrmixzquldialdwrdjmbhv\nfybncihpzjcbxefzvriisgfipsiapdpbssglqjuhvqrxefptim\nzisgnxjhjetomwbttpgbszzuwrfnorzxtomuwnqpkuttcbybdh\njdifyervjaoalcgxovldqfzaorahdigyojknviaztpcmxlvova\nuicnyxqnktylaqnvbjlltexgrhifdneugyszncrwgidcfjgdzk\nyahermudccmueahebyvsakxwseqzduyfezujaffdrsqfseqsdf\ndssgscjyqzwcamnvrilessqpzgqxdlsnclujdsaqbxxyltzlqw\nshnebuhtmqqpmbwseocxeobhdjmkfmzvmkezwomlyzpagrxgxy\nredrydnvlxwvhueaiqgluufbbdtpvtcfhprlyrbvvlntrqzmqa\nknedshbyvenboqueaaiqwexnqgqvjwzfofokvkgdgzhwrtktdb\nrrlltgxdyoaokdparwbanidrfssiuigzduwzejbbnuoiwqlsgt\ndognzbyixybmkdaqlixeutojgtdjltwpxpvumbudckmdxrhqwk\noxonavvxwsvprpymskznabsquwssucxrmywerfpziqdziycnyn\nbnpgphefwpqnrbxwdaippueolngeddtrxpaxxziwphxkeinqsd\nkykrymheywujwmjhnzxmgmkqgsajkykvqjyaplnwkcubvbxdxy\nyczlcgyaqnkgijukiiiucjfueqbmxqnwdcyberszmofrwoclby\nodirlsywlymiwshvenliuomflsyiqacjuajhxmeklfadfvxvmp\nkgqysprovnzfrbajessmybyckdqmsxsrydskoiktyfxjombtwy\ntfduzxgpsaqwcnjiytsrzyfwkrdlabywhcmfgzvcybflkacyhh\ndpersjdqucsauprztyfoiiklhtjsznddftcgelchwbizhesdux\neltnlyboqjegcvuuiilmsbnalvbsfutyarenkewzlpwgqzfnnk\nbvzvmmprwaxisdukvmbvcvzwxynrtksbdzkvagihfgiqmsdejx\nmoadizdsortkabmswcmywnbajxnkvmndbvtcpnjxwmoddntsqt\nckasmblhbdjeojnfwylcqvnbmzxugfsiyjguicgffiwrszzdbz\nexhypsbupxuisdeshltsvfhvjvhnxggarvcdaiiaqadyjjerni\nfhjphvshyfiqqtqbxjjmqngqjhwkcexecmdkmzakbzrkjwqdyu\npidedmvrpxqndcahltgzshjaruuemquxrrvthohcdkuurwurkw\nirrowirgqvzelysohwvjfalsgxgzokbfurvecuxsipqvrwqjos\ntwrbyebfsswbrczwxkltzjgmfuxarydaggojpheusiubpmqjmh\nryhjifcxbywhfutfzmjcdlivnuxmrdfghkqlqzaeeazkoomvpy\nzsuvqwphlhglufbhjgbpkxjxiyeuwmdusfmlxkvqsmwytkjoak\nujlkseszafzjmdtsbyldhylcgkyngvmhneqyjdugofklitxaoy\nluortkqadjcquqdqcsvixgwglwrrmkhdsbebbjvcgzlwtbvasp\nocqmvjospdflivmsrcqtwgxdazunytifhxquasvcgdlozuqzwh\nsmvqubtwsgltvmmzmslqdxqiifzkaqhsxzguseuexlycgubhdn\nthgmdavybzbqgcrgvwalcptutzxsqlkchithcdezsaeflddflg\nkfoqkzsznjyarkuprerivhubpehxmoydakklbdnfhfxamotube\nhusvzmkuedbdkbiqdwpwpssrjjflswufmhclmnstzubtcvqvqs\nhwajcjupuwyuvkorvmiuplwvooojckjmaqyspyacdngdnmuyrs\nhszxgyidgldnlzeqndbvacjqyhfkqvliusqmxyeqyqorzmmjws\nfssrcxarcjflftmwngzonwfsvtukxvbsoubbbpbfwjfngelekp\nluvcextuoxmzgibulszobtmxutcclnyxcdfhznmwhesgezpvxp\nheccrzfnehqeizvicxxxesnssfeuzhsjdhxnddsiuvxanfkpeq\nmacmzildecegshhntfznbbbgxavmpnflhpxylxkyztwbqwpuym\ngpuamkxkszhkbpphykinkezplvfjaqmopodotkrjzrimlvumua\nhvmnfvfjykqmimwxemmahhmzoadapworuvlpqolwbmkeskafyz\nqxqjxqhpfzewpyykzeqjpehsgiqvyezbqwnpyuciobbloxjxuo\nxedahmjqtviqrxnfpoysakxhxrjetydsxixvxyujbvqavxzupq\nuskhsmeicaqqesyyvopepmvdosibrvqoghdeikbpqbfgrufxds\nlzvbozjaraefmlotftnqrjolvuamahndekfdsqcfvmqbocbomj\nwbssranjhaxtjvsxsduozsxloeblcifbyfeuonsyricvccfpkz\nexxdsufcjvrbkesrobosuzumuccgmrsmxztpsiqchfclvzkmvf\nvjzgkpdzfjmcjcmsaaskextlsjrggtycgcqfpoqomrouhjkntq\njyhvgpnfsapufjqfespxbfljgtdgsmfeecqfwfvkweiacditms\neshscnvlmjyiwhagtimrqtffkfgbodlwxirgrmrdhlihyamxmo\nnfaldcqlrycllmccmodlnlbkwvgmdzwapsbzyrwxasqvctbmtb\nqchjkuxpxngebxrmqdgqjsosencrbwknllvucvubyozfmttxtl\nvzllkujidhukzwzcltgqngguftuahalwvjwqncksizgzajkhyj\nsrrnrqavshaszrenhlbzpnpjgqftvwgukjwsekfcgllbzlnvmo\nxscqqnnqmpcpkagzsxmcborwjyqnnousxhsdmklmndntfukmsk\nqrhuhlygensbkpwqbwwcznxeiyoztkmgcvjusurktiehahzrnt\nivgplbcszjhsxeicksxbsejhgmkihtjcqqwxftjswwptmgzptq\napkrwisamnxagzfqsfsxtyxepbpumtlujgxuenmzggjmgiutqo\nrrasikbabwcsrhacznxyudgfpqdpguijawghwvvfogdvtuhmor\nlbyfwcdputedzcwskrpkykqypkslsqnfndigwhjpdqmaclgneb\nzvuughetwkxzuzfvoirareyblwprdnetkyigxyqpzxeckygyxt\narbiaeporgwquijhbuwqvejgcojncyuopfgfptteusxajvqynz\nerapdnumytfobbqcozojivhkngvyystjgsdfnosjlrxpzfenfy\nzxecguclnxeatmlqxsjkyjxifirswzudolgmnljjzjujuayjii\nccervhavvgtcwhlsrwaqotogokhtwgjmfqslvhzpgnsfqhgboe\niowewrvfltajjrdugejyjrkwazogerwkhvgtquxuxhrubfrqyf\ntixbnagkqrzesckrihwqplfpspkqvifbmnqwidbkzqiygwfune\nhnfjnfrvhqomofzkqiptsihalujjxkburwsbdllapwrqcarxml\nmuscnbrlczvfzwmmktusajdhocmprjlndydxrojjahocitarxl\nppgykichjtpukfnlxfcevkjezqsmeycanjlbessrfazdprcomd\nsjwncyqyabutsitezurqhbuwabexrcuiwafnfvcasmrmbqnupr\ntioalhqhqekydzqxmnzcaplzlyxvfbpdzmlsfmmlgtnfraroeb\nzqwfcalvwxjflfjtstvrctlbkbsjpnxyhscxdxepbwqecewrzc\npzdqlnddxfanwnktlpjccpwuygxyyxuafypzoakdxpkahvpjjf\nawgcubspxsiuytqrwmvxfsvzlotlfwimliygnfddeswmxuvhnq\noislcmorsfgzzksposltnxpvohdspmzkewhwnjmmmgiquywbyj\npjimsfbuslksyveergcgmonctcsvypolplcgsqyfkilrixodiw\nypiruisjqibacpbombsjoqozzsdgdrgyqyrfrkssntfgudfqrq\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
