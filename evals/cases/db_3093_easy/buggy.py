from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def convertToTitle(self, n: int) -> str:
        if n < 27:
            return chr(ord("A") + (n - 1) % 26)
        ans = ""
        if n > 0:
            if n % 26 == 0:
                ans += chr(ord("A") + 25)
                n -= 1
            else:
                ans += chr(ord("A") + n % 26 - 1)
            n //= 26
        return ans[::-1]
