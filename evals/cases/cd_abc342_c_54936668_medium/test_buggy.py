import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7\natcoder\n4\nr a\nt e\nd v\na r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'recover\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\nabc\n4\na a\ns k\nn n\nz b\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'abc\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='34\nsupercalifragilisticexpialidocious\n20\ng c\nl g\ng m\nc m\nr o\ns e\na a\no f\nf s\ne t\nt l\nd v\np k\nv h\nx i\nh n\nn j\ni r\ns i\nu a\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'laklimamriiamrmrllrmlrkramrjimrial\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\ngxt\n2002\nt l\nx q\nl x\ng d\nq z\nx u\nu h\nh b\nz f\nd w\nb v\nw x\nv r\nf q\nr n\nx v\nq y\nv c\nn g\ny l\nl w\ng p\nc t\nw t\nt m\np a\na f\nf p\np x\nx a\nm c\na t\nt e\ne c\nc l\nl v\nv z\nz h\nh s\ns b\nb w\nw r\nr e\ne p\np e\ne g\ng v\nv c\nc h\nh w\nw y\ny w\nw f\nf u\nu s\ns f\nf v\nv h\nh s\ns z\nz c\nc j\nj c\nc h\nh m\nm a\na v\nv e\ne q\nq u\nu s\ns l\nl r\nr k\nk e\ne r\nr g\ng a\na s\ns u\nu u\nu u\nu c\nc z\nz a\na z\nz n\nn p\np y\ny a\na i\ni o\no w\nw i\ni n\nn b\nb u\nu a\na n\nn d\nd r\nr f\nf b\nb l\nl z\nz v\nv s\ns s\ns c\nc t\nt d\nd l\nl e\ne h\nh i\ni t\nt b\nb v\nv e\ne j\nj l\nl e\ne m\nm d\nd j\nj w\nw a\na d\nd n\nn o\no i\ni r\nr j\nj r\nr t\nt y\ny o\no o\no x\nx v\nv x\nx d\nd h\nh f\nf p\np s\ns n\nn n\nn v\nv a\na b\nb j\nj c\nc g\ng w\nw k\nk e\ne s\ns y\ny d\nd e\ne l\nl m\nm x\nx m\nm y\ny y\ny x\nx d\nd d\nd m\nm r\nr v\nv f\nf t\nt x\nx j\nj s\ns e\ne d\nd b\nb l\nl z\nz l\nl h\nh b\nb z\nz h\nh q\nq t\nt m\nm u\nu f\nf m\nm z\nz m\nm a\na r\nr s\ns a\na v\nv r\nr f\nf x\nx r\nr w\nw z\nz q\nq d\nd m\nm p\np k\nk s\ns i\ni j\nj r\nr s\ns h\nh c\nc m\nm k\nk h\nh u\nu r\nr u\nu s\ns v\nv o\no e\ne l\nl x\nx p\np u\nu f\nf p\np c\nc v\nv n\nn g\ng z\nz n\nn b\nb t\nt d\nd z\nz t\nt r\nr i\ni u\nu a\na j\nj s\ns f\nf a\na e\ne y\ny e\ne m\nm p\np w\nw f\nf c\nc j\nj u\nu y\ny d\nd p\np j\nj q\nq m\nm p\np i\ni b\nb r\nr a\na h\nh v\nv v\nv r\nr l\nl s\ns t\nt a\na q\nq g\ng z\nz q\nq b\nb x\nx l\nl x\nx k\nk r\nr h\nh e\ne v\nv s\ns i\ni i\ni o\no w\nw i\ni m\nm s\ns s\ns c\nc s\ns i\ni g\ng c\nc q\nq k\nk m\nm t\nt b\nb r\nr b\nb w\nw q\nq h\nh d\nd z\nz m\nm h\nh n\nn e\ne g\ng m\nm p\np z\nz k\nk v\nv a\na w\nw k\nk r\nr u\nu a\na d\nd d\nd a\na o\no c\nc x\nx h\nh n\nn g\ng m\nm w\nw n\nn x\nx w\nw u\nu n\nn i\ni i\ni m\nm d\nd d\nd r\nr f\nf w\nw u\nu b\nb j\nj l\nl k\nk l\nl l\nl a\na t\nt f\nf k\nk r\nr b\nb j\nj c\nc i\ni e\ne y\ny w\nw c\nc l\nl o\no s\ns x\nx c\nc v\nv b\nb f\nf w\nw u\nu v\nv k\nk p\np s\ns j\nj k\nk s\ns f\nf y\ny n\nn e\ne k\nk h\nh f\nf g\ng b\nb w\nw m\nm n\nn i\ni g\ng v\nv a\na n\nn e\ne r\nr j\nj s\ns z\nz b\nb k\nk v\nv b\nb k\nk n\nn a\na e\ne a\na i\ni k\nk h\nh m\nm n\nn c\nc g\ng d\nd f\nf b\nb e\ne d\nd i\ni d\nd p\np m\nm k\nk s\ns b\nb h\nh e\ne n\nn j\nj p\np i\ni v\nv u\nu c\nc q\nq f\nf m\nm y\ny v\nv n\nn c\nc a\na a\na t\nt h\nh h\nh o\no h\nh x\nx u\nu c\nc r\nr w\nw e\ne t\nt f\nf s\ns s\ns x\nx m\nm n\nn v\nv p\np x\nx r\nr r\nr o\no d\nd f\nf o\no d\nd p\np i\ni f\nf w\nw z\nz o\no j\nj e\ne d\nd x\nx p\np y\ny o\no p\np s\ns s\ns v\nv y\ny i\ni s\ns o\no y\ny y\ny b\nb o\no r\nr y\ny w\nw z\nz x\nx z\nz g\ng p\np g\ng r\nr m\nm c\nc z\nz k\nk u\nu w\nw a\na z\nz x\nx u\nu b\nb q\nq b\nb r\nr u\nu i\ni y\ny d\nd l\nl l\nl q\nq f\nf p\np y\ny u\nu h\nh w\nw x\nx g\ng n\nn x\nx j\nj f\nf g\ng p\np a\na h\nh d\nd j\nj m\nm d\nd e\ne a\na r\nr a\na w\nw k\nk m\nm d\nd o\no a\na q\nq x\nx l\nl c\nc t\nt h\nh o\no e\ne g\ng t\nt e\ne b\nb n\nn i\ni v\nv e\ne k\nk a\na e\ne z\nz f\nf b\nb x\nx n\nn y\ny s\ns l\nl v\nv b\nb t\nt b\nb o\no h\nh w\nw x\nx f\nf j\nj u\nu y\ny n\nn q\nq j\nj m\nm z\nz u\nu n\nn i\ni t\nt k\nk t\nt y\ny p\np p\np c\nc c\nc b\nb p\np g\ng n\nn n\nn n\nn x\nx q\nq h\nh o\no l\nl s\ns p\np p\np u\nu i\ni a\na z\nz s\ns g\ng e\ne h\nh f\nf e\ne d\nd f\nf g\ng i\ni k\nk m\nm o\no a\na n\nn b\nb s\ns s\ns g\ng g\ng j\nj a\na d\nd h\nh z\nz c\nc d\nd d\nd f\nf r\nr a\na o\no c\nc a\na x\nx z\nz c\nc d\nd w\nw b\nb p\np i\ni n\nn r\nr b\nb p\np d\nd m\nm f\nf c\nc b\nb b\nb b\nb t\nt v\nv d\nd c\nc r\nr j\nj l\nl l\nl u\nu c\nc t\nt j\nj f\nf p\np p\np r\nr n\nn r\nr f\nf d\nd m\nm j\nj p\np i\ni n\nn i\ni i\ni g\ng r\nr l\nl d\nd y\ny w\nw j\nj b\nb p\np k\nk v\nv c\nc b\nb k\nk s\ns a\na i\ni y\ny i\ni m\nm r\nr q\nq c\nc g\ng i\ni c\nc q\nq t\nt f\nf p\np h\nh b\nb a\na r\nr m\nm f\nf c\nc w\nw e\ne k\nk x\nx o\no g\ng d\nd o\no x\nx g\ng l\nl m\nm j\nj b\nb o\no y\ny x\nx j\nj w\nw z\nz x\nx k\nk c\nc s\ns c\nc d\nd a\na r\nr u\nu c\nc q\nq u\nu t\nt p\np o\no g\ng d\nd o\no c\nc c\nc p\np s\ns x\nx m\nm a\na z\nz w\nw l\nl b\nb u\nu z\nz c\nc j\nj w\nw w\nw f\nf q\nq n\nn r\nr g\ng u\nu m\nm x\nx d\nd b\nb p\np p\np y\ny n\nn t\nt a\na f\nf p\np t\nt i\ni j\nj t\nt i\ni y\ny s\ns h\nh q\nq k\nk t\nt d\nd w\nw o\no w\nw g\ng a\na y\ny f\nf r\nr r\nr s\ns t\nt u\nu y\ny a\na y\ny f\nf u\nu d\nd p\np r\nr u\nu e\ne x\nx l\nl a\na n\nn h\nh c\nc f\nf x\nx r\nr v\nv z\nz k\nk k\nk m\nm f\nf w\nw j\nj j\nj j\nj o\no i\ni k\nk t\nt j\nj f\nf p\np c\nc q\nq r\nr b\nb z\nz a\na v\nv v\nv g\ng g\ng q\nq x\nx q\nq q\nq a\na q\nq g\ng d\nd u\nu x\nx q\nq i\ni k\nk p\np m\nm u\nu v\nv k\nk h\nh o\no g\ng d\nd c\nc j\nj z\nz y\ny s\ns v\nv d\nd x\nx a\na z\nz i\ni o\no f\nf p\np h\nh h\nh f\nf e\ne o\no r\nr q\nq u\nu b\nb l\nl g\ng o\no i\ni h\nh b\nb f\nf d\nd n\nn h\nh a\na a\na s\ns i\ni b\nb h\nh i\ni j\nj t\nt z\nz n\nn o\no k\nk m\nm m\nm x\nx f\nf c\nc x\nx q\nq v\nv o\no m\nm n\nn k\nk n\nn w\nw r\nr v\nv j\nj t\nt d\nd n\nn b\nb b\nb k\nk g\ng g\ng w\nw n\nn o\no t\nt h\nh d\nd u\nu y\ny i\ni n\nn w\nw u\nu o\no f\nf r\nr m\nm g\ng d\nd i\ni i\ni j\nj d\nd s\ns e\ne p\np o\no l\nl g\ng l\nl q\nq y\ny b\nb q\nq y\ny q\nq j\nj i\ni s\ns x\nx r\nr y\ny k\nk z\nz i\ni g\ng k\nk i\ni z\nz r\nr f\nf u\nu u\nu o\no b\nb n\nn v\nv d\nd l\nl j\nj e\ne y\ny s\ns n\nn f\nf w\nw f\nf y\ny y\ny v\nv x\nx a\na u\nu k\nk w\nw g\ng s\ns q\nq c\nc x\nx w\nw c\nc d\nd e\ne d\nd w\nw e\ne l\nl l\nl l\nl e\ne m\nm v\nv o\no q\nq o\no l\nl z\nz z\nz y\ny m\nm p\np d\nd v\nv p\np c\nc e\ne g\ng i\ni p\np o\no a\na q\nq y\ny q\nq f\nf f\nf m\nm h\nh u\nu l\nl l\nl c\nc h\nh v\nv p\np t\nt x\nx h\nh s\ns u\nu v\nv t\nt k\nk r\nr o\no k\nk o\no e\ne d\nd t\nt z\nz m\nm i\ni c\nc v\nv w\nw p\np a\na x\nx m\nm p\np p\np m\nm m\nm g\ng n\nn l\nl w\nw z\nz n\nn i\ni c\nc s\ns i\ni g\ng c\nc w\nw i\ni t\nt a\na j\nj h\nh n\nn u\nu t\nt p\np l\nl z\nz w\nw z\nz j\nj r\nr n\nn i\ni a\na i\ni j\nj e\ne b\nb l\nl v\nv w\nw i\ni s\ns k\nk l\nl m\nm s\ns p\np m\nm h\nh u\nu m\nm r\nr r\nr c\nc x\nx n\nn b\nb v\nv v\nv g\ng i\ni o\no y\ny s\ns a\na q\nq u\nu r\nr a\na k\nk k\nk p\np m\nm l\nl n\nn k\nk o\no e\ne f\nf r\nr g\ng h\nh f\nf l\nl x\nx n\nn x\nx q\nq m\nm f\nf i\ni i\ni k\nk d\nd x\nx m\nm k\nk m\nm r\nr e\ne g\ng e\ne y\ny e\ne q\nq c\nc r\nr g\ng w\nw t\nt b\nb p\np b\nb x\nx d\nd v\nv p\np u\nu d\nd n\nn d\nd b\nb t\nt h\nh v\nv d\nd z\nz w\nw a\na v\nv a\na i\ni p\np j\nj t\nt e\ne i\ni l\nl m\nm g\ng x\nx a\na y\ny g\ng x\nx c\nc h\nh d\nd m\nm d\nd d\nd r\nr h\nh i\ni i\ni q\nq k\nk y\ny z\nz b\nb c\nc n\nn m\nm i\ni t\nt h\nh d\nd q\nq d\nd f\nf d\nd f\nf f\nf w\nw q\nq c\nc x\nx y\ny i\ni a\na f\nf e\ne v\nv i\ni j\nj a\na s\ns r\nr o\no s\ns h\nh l\nl d\nd x\nx x\nx f\nf m\nm p\np m\nm f\nf e\ne j\nj s\ns x\nx n\nn s\ns s\ns k\nk r\nr a\na e\ne k\nk p\np k\nk m\nm g\ng n\nn t\nt o\no a\na h\nh n\nn s\ns w\nw u\nu e\ne f\nf n\nn x\nx j\nj p\np u\nu p\np w\nw i\ni b\nb m\nm q\nq x\nx n\nn j\nj m\nm n\nn g\ng o\no x\nx a\na a\na s\ns d\nd j\nj y\ny e\ne y\ny c\nc s\ns j\nj o\no u\nu v\nv d\nd w\nw n\nn g\ng a\na o\no v\nv m\nm h\nh j\nj y\ny k\nk g\ng f\nf a\na x\nx q\nq u\nu y\ny n\nn r\nr p\np v\nv u\nu f\nf h\nh k\nk v\nv s\ns j\nj c\nc q\nq p\np i\ni q\nq d\nd e\ne z\nz o\no e\ne n\nn q\nq m\nm n\nn n\nn a\na j\nj e\ne d\nd e\ne s\ns s\ns f\nf w\nw t\nt w\nw u\nu x\nx z\nz p\np k\nk p\np x\nx j\nj n\nn n\nn v\nv o\no n\nn w\nw u\nu z\nz y\ny p\np v\nv l\nl v\nv j\nj p\np k\nk g\ng f\nf j\nj c\nc c\nc g\ng c\nc d\nd r\nr w\nw w\nw m\nm n\nn o\no y\ny e\ne u\nu h\nh c\nc o\no s\ns f\nf k\nk b\nb r\nr u\nu p\np x\nx i\ni g\ng j\nj c\nc f\nf v\nv u\nu z\nz e\ne y\ny e\ne s\ns e\ne v\nv y\ny m\nm t\nt y\ny p\np v\nv s\ns k\nk x\nx r\nr g\ng n\nn r\nr r\nr x\nx i\ni i\ni m\nm t\nt l\nl i\ni t\nt h\nh k\nk a\na g\ng s\ns z\nz b\nb k\nk j\nj i\ni i\ni r\nr q\nq l\nl z\nz w\nw e\ne z\nz h\nh j\nj g\ng x\nx n\nn f\nf h\nh a\na z\nz c\nc i\ni t\nt d\nd v\nv q\nq v\nv v\nv d\nd w\nw s\ns b\nb w\nw j\nj i\ni x\nx j\nj o\no n\nn b\nb a\na y\ny v\nv v\nv v\nv q\nq i\ni a\na w\nw x\nx v\nv r\nr g\ng j\nj e\ne b\nb d\nd h\nh u\nu i\ni f\nf u\nu q\nq t\nt z\nz h\nh t\nt s\ns y\ny u\nu a\na w\nw q\nq t\nt d\nd k\nk f\nf j\nj h\nh w\nw r\nr b\nb d\nd q\nq o\no c\nc c\nc p\np s\ns p\np z\nz w\nw w\nw h\nh z\nz w\nw x\nx t\nt y\ny o\no d\nd b\nb o\no q\nq u\nu w\nw v\nv h\nh a\na o\no n\nn u\nu l\nl r\nr u\nu t\nt t\nt x\nx m\nm z\nz n\nn g\ng i\ni u\nu t\nt f\nf a\na v\nv i\ni t\nt o\no r\nr j\nj r\nr y\ny d\nd z\nz a\na e\ne n\nn x\nx u\nu q\nq n\nn n\nn t\nt i\ni l\nl n\nn t\nt n\nn y\ny f\nf y\ny b\nb x\nx c\nc a\na y\ny f\nf n\nn u\nu h\nh u\nu i\ni s\ns g\ng m\nm y\ny m\nm k\nk c\nc z\nz g\ng l\nl a\na k\nk r\nr y\ny h\nh n\nn e\ne m\nm v\nv l\nl l\nl x\nx u\nu v\nv l\nl o\no s\ns j\nj i\ni s\ns u\nu x\nx y\ny c\nc c\nc r\nr i\ni a\na n\nn c\nc p\np r\nr o\no u\nu c\nc o\no m\nm l\nl e\ne j\nj b\nb h\nh k\nk u\nu d\nd b\nb s\ns o\no d\nd n\nn j\nj p\np u\nu y\ny c\nc h\nh d\nd t\nt k\nk s\ns p\np c\nc t\nt s\ns r\nr g\ng j\nj q\nq q\nq h\nh u\nu n\nn o\no v\nv t\nt o\no i\ni c\nc g\ng h\nh u\nu e\ne b\nb n\nn t\nt k\nk e\ne a\na k\nk l\nl p\np o\no b\nb x\nx u\nu d\nd p\np v\nv y\ny q\nq c\nc r\nr k\nk j\nj t\nt n\nn h\nh o\no n\nn q\nq l\nl g\ng g\ng k\nk m\nm e\ne d\nd q\nq w\nw j\nj s\ns l\nl t\nt p\np u\nu g\ng f\nf r\nr q\nq x\nx j\nj i\ni d\nd p\np w\nw i\ni w\nw m\nm d\nd n\nn f\nf h\nh n\nn b\nb s\ns b\nb y\ny j\nj b\nb i\ni x\nx q\nq j\nj b\nb v\nv r\nr f\nf y\ny y\ny v\nv h\nh u\nu u\nu p\np o\no k\nk d\nd v\nv v\nv u\nu z\nz d\nd j\nj d\nd a\na s\ns m\nm l\nl h\nh g\ng p\np p\np i\ni p\np z\nz o\no d\nd v\nv k\nk a\na d\nd d\nd y\ny j\nj t\nt w\nw j\nj m\nm a\na v\nv p\np y\ny k\nk d\nd k\nk s\ns h\nh v\nv a\na s\ns c\nc k\nk x\nx o\no e\ne i\ni m\nm c\nc h\nh i\ni s\ns w\nw p\np v\nv g\ng l\nl c\nc q\nq r\nr j\nj w\nw i\ni m\nm w\nw q\nq f\nf o\no o\no p\np k\nk u\nu y\ny c\nc q\nq j\nj e\ne y\ny e\ne f\nf i\ni a\na y\ny j\nj n\nn f\nf r\nr i\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'iii\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\nyws\n4\ny k\nk h\ns g\ng a\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'hwa\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
