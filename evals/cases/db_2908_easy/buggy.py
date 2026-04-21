from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def firstPalindrome(self, words: ArrayList[str]) -> str:
        for i in words:
            if i == i[::-1]:
                return i
        return ""
