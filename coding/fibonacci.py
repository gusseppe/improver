# filename: fibonacci.py

def print_fibonacci(n):
    a, b = 0, 1
    for i in range(n):
        print(a)
        a, b = b, a + b

print_fibonacci(10)
TERMINATE