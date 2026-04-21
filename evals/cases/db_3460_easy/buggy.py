from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def alternateDigitSum(self, n: int) -> int:
        result = str(n)
        sum = 0
        for i in range(len(result)):
            if i % 2 == 0:
                sum = sum + int(result[i])
            else:
                sum = sum + int(result[i])
        return sum
