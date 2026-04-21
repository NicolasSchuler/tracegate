from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def countGoodNumbers(self, n: int) -> int:
        return (pow(5, (n - 1) // 2, 1000000007) * pow(4, n // 2, 1000000007)) % 1000000007
