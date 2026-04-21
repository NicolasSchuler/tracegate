from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.restoreIpAddresses("25525511135")
    assert result == ["255.255.11.135","255.255.111.35"]


def test_case_1():
    sol = Solution()
    result = sol.restoreIpAddresses("0000")
    assert result == ["0.0.0.0"]


def test_case_2():
    sol = Solution()
    result = sol.restoreIpAddresses("101023")
    assert result == ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
