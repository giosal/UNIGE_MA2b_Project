import csv
train_reader = [];
t = []
with open('train.txt', newline='') as train:
    train_reader = csv.reader(train, delimiter=' ')
    for tr in train_reader:
        print(tr[0])
        t.append(tr[0])

print(t)

res = []
with open('test.txt', newline='') as test:
    test_reader = csv.reader(test, delimiter=' ')
    for te in test_reader:
        #print(te)
        if te[0] in t:

            res.append(te[0])
            print(te[0])

print(res)
test.close()
train.close()
