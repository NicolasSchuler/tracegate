import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 3\nabc\naaaaa\nxyz\na\ndef\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'aaaaa\nabc\nxyz\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 4\nz\nzyx\nzzz\nrbg\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'rbg\nz\nzyx\nzzz\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 1\nabc\narc\nagc\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'abc\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100 100\nwxxihohrzy\nwxxihohrzw\nwxxihohrzu\nwxxihohrzs\nwxxihohrzo\nwxxihohrzn\nwxxihohrzl\nwxxihohrzj\nwxxihohrze\nwxxihohrzb\nwxxihohrty\nwxxihohrtw\nwxxihohrtu\nwxxihohrts\nwxxihohrto\nwxxihohrtn\nwxxihohrtl\nwxxihohrtj\nwxxihohrte\nwxxihohrtb\nwxxihohrqy\nwxxihohrqw\nwxxihohrqu\nwxxihohrqs\nwxxihohrqo\nwxxihohrqn\nwxxihohrql\nwxxihohrqj\nwxxihohrqe\nwxxihohrqb\nwxxihohrpy\nwxxihohrpw\nwxxihohrpu\nwxxihohrps\nwxxihohrpo\nwxxihohrpn\nwxxihohrpl\nwxxihohrpj\nwxxihohrpe\nwxxihohrpb\nwxxihohrmy\nwxxihohrmw\nwxxihohrmu\nwxxihohrms\nwxxihohrmo\nwxxihohrmn\nwxxihohrml\nwxxihohrmj\nwxxihohrme\nwxxihohrmb\nwxxihohriy\nwxxihohriw\nwxxihohriu\nwxxihohris\nwxxihohrio\nwxxihohrin\nwxxihohril\nwxxihohrij\nwxxihohrie\nwxxihohrib\nwxxihohrgy\nwxxihohrgw\nwxxihohrgu\nwxxihohrgs\nwxxihohrgo\nwxxihohrgn\nwxxihohrgl\nwxxihohrgj\nwxxihohrge\nwxxihohrgb\nwxxihohrfy\nwxxihohrfw\nwxxihohrfu\nwxxihohrfs\nwxxihohrfo\nwxxihohrfn\nwxxihohrfl\nwxxihohrfj\nwxxihohrfe\nwxxihohrfb\nwxxihohrdy\nwxxihohrdw\nwxxihohrdu\nwxxihohrds\nwxxihohrdo\nwxxihohrdn\nwxxihohrdl\nwxxihohrdj\nwxxihohrde\nwxxihohrdb\nwxxihohray\nwxxihohraw\nwxxihohrau\nwxxihohras\nwxxihohrao\nwxxihohran\nwxxihohral\nwxxihohraj\nwxxihohrae\nwxxihohrab\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'wxxihohrab\nwxxihohrae\nwxxihohraj\nwxxihohral\nwxxihohran\nwxxihohrao\nwxxihohras\nwxxihohrau\nwxxihohraw\nwxxihohray\nwxxihohrdb\nwxxihohrde\nwxxihohrdj\nwxxihohrdl\nwxxihohrdn\nwxxihohrdo\nwxxihohrds\nwxxihohrdu\nwxxihohrdw\nwxxihohrdy\nwxxihohrfb\nwxxihohrfe\nwxxihohrfj\nwxxihohrfl\nwxxihohrfn\nwxxihohrfo\nwxxihohrfs\nwxxihohrfu\nwxxihohrfw\nwxxihohrfy\nwxxihohrgb\nwxxihohrge\nwxxihohrgj\nwxxihohrgl\nwxxihohrgn\nwxxihohrgo\nwxxihohrgs\nwxxihohrgu\nwxxihohrgw\nwxxihohrgy\nwxxihohrib\nwxxihohrie\nwxxihohrij\nwxxihohril\nwxxihohrin\nwxxihohrio\nwxxihohris\nwxxihohriu\nwxxihohriw\nwxxihohriy\nwxxihohrmb\nwxxihohrme\nwxxihohrmj\nwxxihohrml\nwxxihohrmn\nwxxihohrmo\nwxxihohrms\nwxxihohrmu\nwxxihohrmw\nwxxihohrmy\nwxxihohrpb\nwxxihohrpe\nwxxihohrpj\nwxxihohrpl\nwxxihohrpn\nwxxihohrpo\nwxxihohrps\nwxxihohrpu\nwxxihohrpw\nwxxihohrpy\nwxxihohrqb\nwxxihohrqe\nwxxihohrqj\nwxxihohrql\nwxxihohrqn\nwxxihohrqo\nwxxihohrqs\nwxxihohrqu\nwxxihohrqw\nwxxihohrqy\nwxxihohrtb\nwxxihohrte\nwxxihohrtj\nwxxihohrtl\nwxxihohrtn\nwxxihohrto\nwxxihohrts\nwxxihohrtu\nwxxihohrtw\nwxxihohrty\nwxxihohrzb\nwxxihohrze\nwxxihohrzj\nwxxihohrzl\nwxxihohrzn\nwxxihohrzo\nwxxihohrzs\nwxxihohrzu\nwxxihohrzw\nwxxihohrzy\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='26 1\ns\nf\nn\nv\na\ng\ny\nj\ni\nh\nw\nd\nb\nk\nq\nm\nu\nr\nc\no\nx\nl\ne\np\nz\nt\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 's\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
