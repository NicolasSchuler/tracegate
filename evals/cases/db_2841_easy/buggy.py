from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
class Solution:
def canWinNim(self, n: int) -> bool:
    return n % 4 != 0
