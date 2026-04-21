from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def alternateDigitSum(self, n: int) -> int:
        count = 0
        n = str(n)
        for i in range(len(n)):
            if i % 2 == 0:
                count += int(n[i])
            else:
                count += int(n[i])
        return count
