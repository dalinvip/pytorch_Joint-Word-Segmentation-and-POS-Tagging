# @Author : bamtercelboo
# @Datetime : 2018/8/23 14:17
# @File : pickle_test.py
# @Last Modify Time : 2018/8/23 14:17
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  pickle_test.py
    FUNCTION : None
"""

import pickle

from Pickle import pcl


class P(object):
    def __init__(self):
        # print("pickle")
        self.word2id = {"I": 1, "love": 2, "Beijing": 3}
        self.id2word = [1, 2, 3]
        self.mincount = 3


w = P()
l = P()
d = [w, l, 1, 2, 3, 4]
print(w, l)
path = "../alphabet.pkl"
# obj_dict = {"w": w, "l": l, "d": d}
# pcl.save(obj=w, path=path)
data = pcl.load(path)
print(data)
# print(data["w"].word2id)
# print(data["l"].id2word)
# print(data["d"])

# output = open('alphabet.pkl', 'wb')
# # output = open('alphabet.txt', 'w')
# # pickle.dumps(w)
# # pickle.loads(w)
# # print(w.word2id)
# pickle.sa(w, output)
# # pickle.dump(l, output, -1)
# # pickle.dump(d, output, -1)
# output.close()
