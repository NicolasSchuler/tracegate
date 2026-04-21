from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools

# Import RegEx
import re
from collections import Counter


class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        # List words in paragraph, replacing punctuation with ' ' and all lower case
        paragraph = re.subn("[.,!?;']", " ", paragraph.lower())[0].split(" ")

        # Remove any '' or words in banned from paragraph list
        paragraph = list(filter(lambda x: x in banned + [""], paragraph))

        # Return most common word in filtered list
        return Counter(paragraph).most_common(1)[0][0]
