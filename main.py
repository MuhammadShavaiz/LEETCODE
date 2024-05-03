from collections import defaultdict
from typing import List

"""
Contains Duplicate:
Given an integer array nums, return true if any value appears at least twice in the array,
 and return false if every element is distinct.
Example 1:
Input: nums = [1,2,3,1]
Output: true
"""
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False

"""
Valid Anagram:
Given two strings s and t, return true if t is an anagram of s, and false otherwise.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase,
typically using all the original letters exactly once.
Example 1:
Input: s = "anagram", t = "nagaram"
Output: true
"""
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        seen = defaultdict(int)
        for i in s:
            seen[i] += 1
        for j in t:
            seen[j] -= 1
        for val in seen.values():
            if val != 0:
                return False
        return True

"""
Two Sum:
Given an array of integers nums and an integer target, return indices of the two numbers 
such that they add up to target. You may assume that each input would have exactly one solution,
 and you may not use the same element twice.
You can return the answer in any order.
Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
"""
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}
        for i in range(len(nums)):
            seen[nums[i]] = i
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in seen and seen[complement] != i:
                return [i, seen[complement]]





