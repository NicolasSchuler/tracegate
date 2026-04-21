from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
class Solution:
    def maxNiceDivisors(self, primeFactors: int) -> int:
        if primeFactors <= 3:
            return primeFactors
        
        MOD = int(1e9 + 7)
        if primeFactors % 3 == 0:
            power = primeFactors // 3
            return self.calculateNiceDivisors(3, power, MOD)
        elif primeFactors % 3 == 1:
            power = (primeFactors // 3) - 1
            return (self.calculateNiceDivisors(3, power, MOD) * 4) % MOD
    else:
        power = primeFactors // 3
        return (self.calculateNiceDivisors(3, power, MOD) * 2) % MOD
    
    def calculateNiceDivisors(self, base: int, power: int, MOD: int) -> int:
        if power == 0:
            return 1

        result = self.calculateNiceDivisors(base, power // 2, MOD)
        result = (result * result) % MOD

        if power % 2 == 1:
            result = (result * base) % MOD

        return result
