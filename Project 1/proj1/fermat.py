import random
import math


def prime_test(N, k):
    # This is main function, that is connected to the Test button. You don't need to touch it.
    return fermat(N, k), miller_rabin(N, k)


def isEven(n):  # Helper function to identify even numbers
    if n % 2 == 0:  # O(n^2) because of division in order to check for even numbers
        return True
    else:
        return False


def mod_exp(x, y, N):
    if y == 0:
        return 1  # O(1)
    z = mod_exp(x, math.floor(y/2), N)  # O(n^2)
    if isEven(y):
        return z**2 % N  # O(n^2)
    else:
        return x * (z**2) % N  # O(n^2)

# Total running time for the function above is O(n^3)


def fprobability(k):
    # You will need to implement this function and change the return value.
    return 1 - (1 / 2**k)
    # runtime is O(n^2) - substraction is O(n),
    # division is O(n^2), exponentiation is O(n)


def mprobability(k):
    # You will need to implement this function and change the return value.
    return 1 - (1 / 4**k)
    # runtime is O(n^2) - substraction is O(n),
    # division is O(n^2), exponentiation is O(n)


def fermat(N, k):
    # You will need to implement this function and change the return value, which should be
    # either 'prime' or 'composite'.
    #
    # To generate random values for a, you will most likley want to use
    # random.randint(low,hi) which gives a random integer between low and
    #  hi, inclusive.

    low = 2
    high = N - 1
    for _ in range(0, k):
        a = random.randint(low, high)
        if mod_exp(a, N - 1, N) == 1:  # O(n^3)
            continue
        else:
            return 'composite'  # O(1)
    return 'prime'  # O(1)
    # This function will run on O(k * n^3). O(n^3) because of the
    # mod_exp function, multiplied by k because of the range


def miller_rabin(N, k):
    # You will need to implement this function and change the return value, which should be
    # either 'prime' or 'composite'.
    #
    # To generate random values for a, you will most likley want to use
    # random.randint(low,hi) which gives a random integer between low and
    #  hi, inclusive.

    low = 2
    high = N - 1
    for _ in range(0, k):
        p = (N - 1) * 2  # O(n)
        a = random.randint(low, high)

        while p > 1 and isEven(p):
            p = p / 2
            mod = mod_exp(a, p, N)  # O(n^3)
            if mod != 1:
                if mod == N - 1:
                    break
                else:
                    return 'composite'
    return 'prime'
    # This function will run on O(k * n^4). As mod_exp runs on
    # O(n^3), after accounting for that, we also account for the divisions and
    # multiplications, getting O(n^4). Also, the main loop will range up to k, thus
    # we also multiply k into the runtime.
