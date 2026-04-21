from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
class Solution:
def distributeCandies(self, candyType: List[int]) -> int:
        return min(len(candyType)//2, len(set(candyType)))
