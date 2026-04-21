from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        dp = defaultdict(Counter)
        for i in range(1, n):
            for j in range(i):
                d = nums[i] - nums[j]
                dp[i][d] += dp[j][d]
            ans += sum(dp[i].values()) - i
        return ans
