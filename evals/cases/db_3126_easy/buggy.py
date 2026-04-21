from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        nums, sums, res, n = sorted(nums), [0], [], len(queries) + 1
        for num in nums:
            sums.append(sums[-1] + num)

        def bSearch(q, l=0, mid=n >> 1, r=n):
            while l < mid < r:
                if sums[mid] > q:
                    r = mid
                elif sums[mid] < q:
                    l = mid
                else:
                    break
                mid = (l + r) >> 1
            return mid

        return [bSearch(q) for q in queries]
