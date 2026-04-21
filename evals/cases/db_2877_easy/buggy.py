from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
class Solution:
def getNoZeroIntegers(self, n: int) -> List[int]:
    for i in range(n-1, -1, -1):
        if '0' not in str(i) and '0' not in str(n - i):
            return [i, n-i ]

    

    return next([i, n-i] for i in range(n-1, -1, -1) if '0' not in str(i) and '0' not in str(n-i))
