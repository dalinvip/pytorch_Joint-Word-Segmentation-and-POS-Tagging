# @Author : bamtercelboo
# @Datetime : 2018/8/23 12:26
# @File : Pickle.py
# @Last Modify Time : 2018/8/23 12:26
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Pickle.py
    FUNCTION : None
"""

# Introduce python packages
import sys
import os
import pickle

# Introduce missing packages in here


class Pickle(object):
    def __init__(self):
        print("Pickle")
        self.obj_count = 0

    @staticmethod
    def save(obj, path, mode="wb"):
        """
        :param obj:  obj dict to dump
        :param path: save path
        :param mode:  file mode
        """
        print("save obj to {}".format(path))
        # print("obj", obj)
        assert isinstance(obj, dict), "The type of obj must be a dict type."
        if os.path.exists(path):
            os.remove(path)
        pkl_file = open(path, mode=mode)
        pickle.dump(obj, pkl_file)
        pkl_file.close()

    @staticmethod
    def load(path, mode="rb"):
        """
        :param path:  pkl path
        :param mode: file mode
        :return: data dict
        """
        print("load obj from {}".format(path))
        if os.path.exists(path) is False:
            print("Path {} illegal.".format(path))
        pkl_file = open(path, mode=mode)
        data = pickle.load(pkl_file)
        pkl_file.close()
        return data


pcl = Pickle




