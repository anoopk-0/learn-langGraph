# Lambda Functions in Python
# --------------------------
# Lambda functions are small, anonymous functions defined with the `lambda` keyword.
# They are useful for short operations, especially as arguments to functions like map, filter, and sorted.
# Syntax: lambda arguments: expression
#
# Example 1: Squaring numbers in a list using map and a lambda

square = lambda x: x * x

nums = [1, 2, 3, 4, 5]
squared_nums = list(map(square, nums))
print(squared_nums)  # Output: [1, 4, 9, 16, 25]

# Example 2: Filtering even numbers using filter and a lambda
even_nums = list(filter(lambda x: x % 2 == 0, nums))
print(even_nums)  # Output: [2, 4]

# Example 3: Sorting tuples by the second element using a lambda as key
pairs = [(1, 3), (2, 2), (4, 1)]
sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
print(sorted_pairs)  # Output: [(4, 1), (2, 2), (1, 3)]
