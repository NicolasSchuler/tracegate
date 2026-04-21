class Solution:
    def minimumOperations(self, nums: list[int]) -> int:
        uniq_non_zero = set()
        for num in nums:
            if num == 0:
                continue
            uniq_non_zero.add(num)
        return len(uniqueCount(uniq_non_zero))

    def uniqueCount(self, num_set: set) -> int:
        return len(num_set)
