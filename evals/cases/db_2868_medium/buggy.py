from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
class Solution:
def wordSubsets(self, words1: List[str], words2: List[str]) -> List[str]:
    w2 = reduce(operator.or_, map(Counter, words2))
    return [w1 for w1 in words1 if Counter(w1) >= w2]
