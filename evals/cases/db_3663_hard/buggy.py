from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def maxPower(self, stations: List[int], r: int, k: int) -> int:
        (start,)
        while start + 1 < end:
            mid = (start + end) // 2
            if self.check(stations, r, k, mid):
                start = mid
            else:
                end = mid
        if self.check(stations, r, k, end):
            return end
        else:
            return start

    def check(self, stations, r, k, target):
        n = len(stations)
        ans = True
        newStations = defaultdict(int)
        power = sum(stations[:r])
        for i in range(n + 1):
            if i + r < n:
                power += stations[i + r]
            if i - r - 1 >= 0:
                power -= stations[i - r - 1]
            if power >= target:
                continue
            elif power + k < target:
                ans = False
                break
            else:
                diff = target - power
                power = target
                stations[min(i + r, n - 1)] -= diff
                k -= diff
                newStations[min(i + r, n - 1)] += diff
        for i in newStations:
            stations[i] -= newStations[i]
        return ans


my_solution = Solution()
my_solution.maxPower([1, 2, 3], 2, 5)
