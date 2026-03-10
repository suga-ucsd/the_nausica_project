import random

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr.pop(0)
    left =
    mid =
    right =
    return quicksort(left) + mid + quicksort(right)

# Example usage:
random.seed(42)
data = [3, 10, 5, 7, 9, 2, 8, 4, 6, 1]
print(f"Original data:\n{sorted(data)}")

