from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def lexicalOrder(self, n: int) -> List[int]:
        lst = [str(i) for i in range(1, n)]
        lst.sort()
        return [int(i) for i in lst]
