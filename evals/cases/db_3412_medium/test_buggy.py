from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.findAllRecipes(["bread"], [["yeast","flour"]], ["yeast","flour","corn"])
    assert result == ["bread"]


def test_case_1():
    sol = Solution()
    result = sol.findAllRecipes(["bread","sandwich"], [["yeast","flour"],["bread","meat"]], ["yeast","flour","meat"])
    assert result == ["bread","sandwich"]


def test_case_2():
    sol = Solution()
    result = sol.findAllRecipes(["bread","sandwich","burger"], [["yeast","flour"],["bread","meat"],["sandwich","meat","bread"]], ["yeast","flour","meat"])
    assert result == ["bread","sandwich","burger"]
