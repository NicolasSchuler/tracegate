from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.partitionLabels("ababcbacadefegdehijhklij")
    assert result == [9,7,8]


def test_case_1():
    sol = Solution()
    result = sol.partitionLabels("eccbbbbdec")
    assert result == [10]
