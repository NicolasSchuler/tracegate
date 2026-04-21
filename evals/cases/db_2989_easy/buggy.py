from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def sortString(self, s: str) -> str:
        freq = {}
        letters = sorted(set(s))
        res = ""
        for i in s:
            if i in freq:
                freq[i] += 1
            else:
                freq[i] = 1
        while freq:
            for i in letters:
                if i in freq:
                    if freq[i] > 0:
                        res += i
                        freq[i] -= 1
                    else:
                        del freq[i]
                        self.undefinedMethod()

            for i in letters[::-1]:
                if i in freq:
                    if freq[i] > 0:
                        res += i
                        freq[i] -= 1
                    else:
                        del freq[i]
                        self.undefinedMethod()

        return res
