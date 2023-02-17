"""
author: nabin

"""
import math
number = list()
score = list()
pred = list()


with open("output.txt", "r") as f:
    lines = f.readlines()
    for i in range(len(lines)):
        val = lines[i]
        key, value = val.split()
        number.append(int(key))
        score.append(float(value))
a = zip(number, score)
b = list(a)
res = sorted(b, key=lambda x: x[0])
for i in res:
    pred.append(i[1])
total = 0.0
original = list()
tmp = list()

file = open("ua.base.feature", "r")
for line in file:
    original.append(float(line[0]))
for i in range(len(original)):
    val_o = (original[i])
    val_p = (pred[i])
    value = math.pow((val_o - val_p),2)
    total += value
rmse = math.sqrt(1 / len(original) * total)
print("The Root Mean Squared Error (RMSE) of mrjob program is", rmse)

