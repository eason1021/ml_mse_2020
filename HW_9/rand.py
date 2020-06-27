import random
list_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
list_2 = list_1.copy()
list_3 = []
for i in range(len(list_1)) :
    ran = random.randint(0, len(list_2) - 1)
    list_3.append(list_2.pop(ran))
print(list_3)