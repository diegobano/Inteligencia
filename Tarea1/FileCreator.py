import numpy as np
import random

db = np.zeros((1151, 17))
f = open("Base_de_datos_tarea_1.txt", "r")
for (i, line) in enumerate(f):
    db[i] = line.split(",")

db_pa = np.mean(db[:, 16])

random.shuffle(db)
ex = db[0:(len(db)/10)*8,:]
ex_pa = np.mean(ex[:, 16])
test = db[(len(db)/10)*8:, :]
test_pa = np.mean(test[:, 16])

acc = 0.02

while True:
    print db_pa, test_pa, ex_pa
    if abs(db_pa - ex_pa) < acc and abs(db_pa - test_pa) < acc:
        break;
    random.shuffle(db)
    ex = db[0:(len(db) / 10) * 8, :]
    ex_pa = np.mean(ex[:, 16])
    test = db[(len(db) / 10) * 8:, :]
    test_pa = np.mean(test[:, 16])
f.close()

ex_file = open("examples.txt", "w")
for i in ex:
    for el in range(16):
        ex_file.write(str(i[el]))
        ex_file.write(",")
    ex_file.write(str(i[16]))
    ex_file.write("\n")
ex_file.close()

test_file = open("testing.txt", "w")
for i in test:
    for el in range(16):
        test_file.write(str(i[el]))
        test_file.write(",")
    test_file.write(str(i[16]))
    test_file.write("\n")
test_file.close()