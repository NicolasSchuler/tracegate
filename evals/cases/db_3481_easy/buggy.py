from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def largestInteger(self, num: int) -> int:
        evenlist = []
        oddlist = []
        nums = [int(x) for x in str(num)]
        for i in nums:
            if i % 2 == 0:
                evenlist.append(i)
            else:
                oddlist.append(i)
        even = [-x for x in evenlist]
        odd = [-x for x in oddlist]
        result = []
        for ele in nums:
            if ele in evenlist:
                result += [-heapq.heappop(even)]
            if ele in oddlist:
                result += [-heapq.heappop(odd)]
        result = [str(x) for x in result]
        return int("".join(result))


s = Solution()
print(s.largestInteger(1234))
