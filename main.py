# @Author : bamtercelboo
# @Datetime : 2018/1/30 19:50
# @File : main.py
# @Last Modify Time : 2018/1/30 19:50
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  main.py
    FUNCTION : main
"""

import argparse
import datetime
import Config.config as configurable
from DataUtils.mainHelp import *
from DataUtils.Alphabet import *
from test import load_test_data
from test import T_Inference
from trainer import Train
import random

# solve default encoding problem
from imp import reload
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
torch.manual_seed(seed_num)
random.seed(seed_num)


def start_train(train_iter, dev_iter, test_iter, model, config):
    """
    :param train_iter:  train batch data iterator
    :param dev_iter:  dev batch data iterator
    :param test_iter:  test batch data iterator
    :param model:  nn model
    :param config:  config
    :return:  None
    """
    t = Train(train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter, model=model, config=config)
    t.train()
    print("Finish Train.")


def start_test(train_iter, dev_iter, test_iter, model, alphabet, config):
    """
    :param train_iter:  train batch data iterator
    :param dev_iter:  dev batch data iterator
    :param test_iter:  test batch data iterator
    :param model:  nn model
    :param alphabet:  alphabet dict
    :param config:  config
    :return:  None
    """
    print("Sorry For Test, Updating......")
    exit()


def main():
    """
    main()
    :return:
    """
    # save file
    config.mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # config.add_args(key="mulu", value=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    config.save_dir = os.path.join(config.save_direction, config.mulu)
    if not os.path.isdir(config.save_dir): os.makedirs(config.save_dir)

    # get data, iter, alphabet
    train_iter, dev_iter, test_iter, alphabet, alphabet_static = load_data(config=config)

    # get params
    get_params(config=config, alphabet=alphabet, alphabet_static=alphabet_static)

    # save dictionary
    save_dictionary(config=config)
    # exit()

    model = load_model(config)

    # print("Training Start......")
    if config.train is True:
        start_train(train_iter, dev_iter, test_iter, model, config)
        exit()
    elif config.test is True:
        start_test(train_iter, dev_iter, test_iter, model, alphabet, config)
        exit()


def parse_argument():
    """
    :argument
    :return:
    """
    parser = argparse.ArgumentParser(description="NER & POS")
    parser.add_argument("-c", "--config", dest="config_file", type=str, default="./Config/config.cfg",
                        help="config path")
    parser.add_argument("-device", "--device", dest="device", type=str, default="cuda:0",
                        help="device[‘cpu’,‘cuda:0’,‘cuda:1’,......]")
    parser.add_argument("--train", dest="train", action="store_true", default=True, help="train model")
    parser.add_argument("-p", "--process", dest="process", action="store_true", default=True, help="data process")
    parser.add_argument("-t", "--test", dest="test", action="store_true", default=False, help="test model")
    parser.add_argument("--t_model", dest="t_model", type=str, default=None, help="model for test")
    parser.add_argument("--t_data", dest="t_data", type=str, default=None,
                        help="data[train dev test None] for test model")
    parser.add_argument("--predict", dest="predict", action="store_true", default=False, help="predict model")
    args = parser.parse_args()
    # print(vars(args))
    config = configurable.Configurable(config_file=args.config_file)
    config.device = args.device
    config.train = args.train
    config.process = args.process
    config.test = args.test
    config.t_model = args.t_model
    config.t_data = args.t_data
    config.predict = args.predict
    # config
    if config.test is True:
        config.train = False
    if config.t_data not in [None, "train", "dev", "test"]:
        print("\nUsage")
        parser.print_help()
        print("t_data : {}, not in [None, 'train', 'dev', 'test']".format(config.t_data))
        exit()
    print("***************************************")
    print("Device : {}".format(config.device))
    print("Data Process : {}".format(config.process))
    print("Train model : {}".format(config.train))
    print("Test model : {}".format(config.test))
    print("t_model : {}".format(config.t_model))
    print("t_data : {}".format(config.t_data))
    print("predict : {}".format(config.predict))
    print("***************************************")

    return config


if __name__ == "__main__":

    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    config = parse_argument()
    if config.device != cpu_device:
        print("Using GPU To Train......")
        device_number = config.device[-1]
        torch.cuda.set_device(int(device_number))
        print("Current Cuda Device {}".format(torch.cuda.current_device()))
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        print("torch.cuda.initial_seed", torch.cuda.initial_seed())

    main()

