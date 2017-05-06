import numpy as np
import random

db = np.zeros((5500, 49))
f = open("sensorless_tarea2.txt", "r")
for (i, line) in enumerate(f):
    db[i] = line.split(",")
f.close()

db_pbs = [0] * 11
for i in range(len(db)):
    db_pbs[int(db[i, 48])-1] += 1

random.shuffle(db)

ex = db[0:(len(db)/10)*8,:]
ex_pbs = [0] * 11
test = db[(len(db)/10)*8:, :]
test_pbs = [0] * 11
for i in range(len(ex)):
    ex_pbs[int(ex[i, 48])-1] += 1
for i in range(len(test)):
    test_pbs[int(test[i, 48])-1] += 1

acc = 0.015
done = True

while True:
    for i in range(11):

        if abs(float(db_pbs[i])/sum(db_pbs) - float(ex_pbs[i])/sum(ex_pbs)) > acc or\
                        abs(float(db_pbs[i]) / sum(db_pbs) - float(test_pbs[i]) / sum(test_pbs)) > acc:
            done = False
            break
        done = True
    if done:
        break
    print len(db)
    random.shuffle(db)
    ex = db[0:(len(db) / 10) * 8, :]
    ex_pbs = [0] * 11
    test = db[(len(db) / 10) * 8:, :]
    test_pbs = [0] * 11
    for i in range(len(ex)):
        ex_pbs[int(ex[i, 48]) - 1] += 1
    for i in range(len(test)):
        test_pbs[int(test[i, 48]) - 1] += 1

print db_pbs
print ex_pbs
print test_pbs


ex_file = open("training.txt", "w")
for i in ex:
    for el in range(48):
        ex_file.write(str(i[el]))
        ex_file.write(",")
    ex_file.write(str(i[48]))
    ex_file.write("\n")
ex_file.close()

test_file = open("tester.txt", "w")
for i in test:
    for el in range(48):
        test_file.write(str(i[el]))
        test_file.write(",")
    test_file.write(str(i[48]))
    test_file.write("\n")
test_file.close()