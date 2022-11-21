import torch
import torch.nn as nn

torch.load('mtface_age_fast.bin')

print()

# class MyNet(nn.Module):
#     def __init__(self):
#         super(MyNet, self).__init__()
#         self.fc1 = nn.Linear(10, 10)
#         self.fc2 = nn.Linear(10, 10)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x
#
#
# def fib():
#     a, b = 0, 1
#     while True:
#         yield a
#         a, b = b, a + b
#
# fib_gen = fib()
# for _ in range(10):
#     next(fib_gen)


import datetime


def parse_expenses(expenses_string):
    """Parse the list of expenses and return the list of triples (date, value, currency).
    Ignore lines starting with #.
    Parse the date using datetime.
    Example expenses_string:
        2016-01-02 -34.01 USD
        2016-01-03  2.59 DKK
        2016-01-03 -2.72 EUR
    """
    expenses = []
    for line in expenses_string.splitlines():
        if line.startswith('#'):
            continue    # ignore comments
        date, value, currency = line.split()
        expenses.append((datetime.datetime.strptime(date, '%Y-%m-%d'), float(value), currency))
    return expenses


def parse_expenses(expenses_string):
    """Parse the set of expenses and return the set of triples (date, value, currency).
    Ignore lines starting with #.
    Parse the date using datetime.
    """
    expenses = []
    for line in expenses_string.splitlines():
        if line.startswith('#'):
            continue    # ignore comments
        date, value, currency = line.split()
        expenses.append((datetime.datetime.strptime(date, '%Y-%m-%d'), float(value), currency))
    return expenses


def generate_pwd(length=8):
    # generate random string, no digits
    import random
    import string
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
