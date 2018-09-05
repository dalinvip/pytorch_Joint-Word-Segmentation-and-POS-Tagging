# @Author : bamtercelboo
# @Datetime : 2018/9/3 10:50
# @File : mainHelp.py
# @Last Modify Time : 2018/9/3 10:50
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  mainHelp.py
    FUNCTION : None
"""

import shutil
from DataUtils.Alphabet import *
from DataUtils.Batch_Iterator import *
from DataUtils.Pickle import pcl
from DataUtils.Embed import Embed
from Dataloader.DataLoader import DataLoader
from models.JointPS import JointPS
from test import load_test_model

# solve default encoding problem
from imp import reload
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
torch.manual_seed(seed_num)
random.seed(seed_num)


def get_learning_algorithm(config):
    """
    :param config:  config
    :return:  optimizer algorithm
    """
    algorithm = None
    if config.adam is True:
        algorithm = "Adam"
    elif config.sgd is True:
        algorithm = "SGD"
    print("learning algorithm is {}.".format(algorithm))
    return algorithm


def get_params(config, alphabet, alphabet_static):
    """
    :param config: config
    :param alphabet:  alphabet dict
    :return:
    """
    # get algorithm
    config.learning_algorithm = get_learning_algorithm(config)

    # save best model path
    config.save_best_model_path = config.save_best_model_dir
    if config.test is False:
        if os.path.exists(config.save_best_model_path):
            shutil.rmtree(config.save_best_model_path)

    # get params
    config.embed_char_num = alphabet.char_alphabet.vocab_size
    config.embed_bichar_num = alphabet.bichar_alphabet.vocab_size
    config.static_embed_char_num = alphabet_static.char_alphabet.vocab_size
    config.static_embed_bichar_num = alphabet_static.bichar_alphabet.vocab_size
    config.label_size = alphabet.label_alphabet.vocab_size
    config.pos_size = alphabet.pos_alphabet.vocab_size

    config.char_paddingId = alphabet.char_PaddingID
    config.bichar_paddingId = alphabet.bichar_PaddingID
    # config.static_char_paddingId = alphabet_static.char_PaddingID
    # config.static_bichar_paddingId = alphabet_static.bichar_PaddingID
    config.create_alphabet = alphabet
    config.create_alphabet_static = alphabet_static
    print("embed_char_num : {}, embed_bichar_num : {}".format(config.embed_char_num, config.embed_bichar_num))
    print("static_embed_char_num : {}, embed_bichar_num : {}".format(config.static_embed_char_num, config.static_embed_bichar_num))
    print("label_size : {}, pos_size : {}".format(config.label_size, config.pos_size))
    print("char_paddingId : {}, bichar_paddingId : {}".format(config.char_paddingId, config.bichar_paddingId))


def save_dict2file(dict, path):
    """
    :param dict:  dict
    :param path:  path to save dict
    :return:
    """
    print("Saving dictionary")
    if os.path.exists(path):
        print("path {} is exist, deleted.".format(path))
    file = open(path, encoding="UTF-8", mode="w")
    for word, index in dict.items():
        # print(word, index)
        file.write(str(word) + "\t" + str(index) + "\n")
    file.close()
    print("Save dictionary finished.")


def save_dictionary(config):
    """
    :param config: config
    :return:
    """
    if config.save_dict is True:
        if os.path.exists(config.dict_directory):
            shutil.rmtree(config.dict_directory)
        if not os.path.isdir(config.dict_directory):
            os.makedirs(config.dict_directory)

        config.word_dict_path = "/".join([config.dict_directory, config.word_dict])
        config.char_dict_path = config.word_dict_path + "_char.txt"
        config.bichar_dict_path = config.word_dict_path + "_bichar.txt"
        config.static_char_dict_path = config.word_dict_path + "_static_char.txt"
        config.static_bichar_dict_path = config.word_dict_path + "_static_bichar.txt"
        config.label_dict_path = "/".join([config.dict_directory, config.label_dict])
        config.pos_dict_path = config.label_dict_path + "_pos.txt"
        config.seg_pos_dict_path = config.label_dict_path + "_seg_pos.txt"
        print("char_dict_path : {}".format(config.char_dict_path))
        print("bichar_dict_path : {}".format(config.bichar_dict_path))
        print("static_char_dict_path : {}".format(config.static_char_dict_path))
        print("static_bichar_dict_path : {}".format(config.static_bichar_dict_path))
        print("pos_dict_path : {}".format(config.pos_dict_path))
        print("seg_pos_dict_path : {}".format(config.seg_pos_dict_path))
        save_dict2file(config.create_alphabet.char_alphabet.words2id, config.char_dict_path)
        save_dict2file(config.create_alphabet.bichar_alphabet.words2id, config.bichar_dict_path)
        save_dict2file(config.create_alphabet_static.char_alphabet.words2id, config.static_char_dict_path)
        save_dict2file(config.create_alphabet_static.bichar_alphabet.words2id, config.static_bichar_dict_path)
        save_dict2file(config.create_alphabet.pos_alphabet.words2id, config.pos_dict_path)
        save_dict2file(config.create_alphabet.label_alphabet.words2id, config.seg_pos_dict_path)
        # copy to mulu
        print("copy dictionary to {}".format(config.save_dir))
        shutil.copytree(config.dict_directory, "/".join([config.save_dir, config.dict_directory]))


# load data / create alphabet / create iterator
def preprocessing(config):
    """
    :param config: config
    :return:
    """
    print("Processing Data......")
    # read file
    data_loader = DataLoader(path=[config.train_file, config.dev_file, config.test_file], shuffle=True, config=config)
    train_data, dev_data, test_data = data_loader.dataLoader()
    print("Train Sentence {}, Dev Sentence {}, Test Sentence {}.".format(len(train_data), len(dev_data), len(test_data)))
    data_dict = {"train_data": train_data, "dev_data": dev_data, "test_data": test_data}
    pcl.save(obj=data_dict, path=os.path.join(config.pkl_directory, config.pkl_data))

    # create the alphabet, alphabet_static
    alphabet = CreateAlphabet(min_freq=config.min_freq, word_min_freq=config.word_min_freq,
                              char_min_freq=config.char_min_freq, bichar_min_freq=config.bichar_min_freq,
                              train_data=train_data, config=config)
    alphabet_static = CreateAlphabet(min_freq=config.min_freq, word_min_freq=config.min_freq,
                                     char_min_freq=config.min_freq, bichar_min_freq=config.min_freq,
                                     train_data=train_data, dev_data=dev_data, test_data=test_data, config=config)
    alphabet.build_vocab()
    alphabet_static.build_vocab()
    alphabet_dict = {"alphabet": alphabet, "alphabet_static": alphabet_static}
    pcl.save(obj=alphabet_dict, path=os.path.join(config.pkl_directory, config.pkl_alphabet))

    # create iterator
    create_iter = Iterators(batch_size=[config.batch_size, config.dev_batch_size, config.test_batch_size],
                            data=[train_data, dev_data, test_data], operator=alphabet, operator_static=alphabet_static,
                            config=config)
    train_iter, dev_iter, test_iter = create_iter.createIterator()
    iter_dict = {"train_iter": train_iter, "dev_iter": dev_iter, "test_iter": test_iter}
    pcl.save(obj=iter_dict, path=os.path.join(config.pkl_directory, config.pkl_iter))
    return train_iter, dev_iter, test_iter, alphabet, alphabet_static


def pre_embed(config, alphabet, alphabet_static):
    """
    :param alphabet_static:
    :param config: config
    :param alphabet:  alphabet dict
    :return:  pre-train embed
    """
    print("***************************************")
    char_pretrain_embed, bichar_pretrain_embed = None, None
    embed_types = ""
    if (config.char_pretrained_embed is True or config.bichar_pretrained_embed is True) and config.zeros:
        embed_types = "zero"
    elif (config.char_pretrained_embed is True or config.bichar_pretrained_embed is True) and config.avg:
        embed_types = "avg"
    elif (config.char_pretrained_embed is True or config.bichar_pretrained_embed is True) and config.uniform:
        embed_types = "uniform"
    elif (config.char_pretrained_embed is True or config.bichar_pretrained_embed is True) and config.nnembed:
        embed_types = "nn"
    if config.char_pretrained_embed is True:
        p = Embed(path=config.char_pretrained_embed_file, words_dict=alphabet_static.word_alphabet.id2words,
                  embed_type=embed_types,
                  pad=paddingkey)
        char_pretrain_embed = p.get_embed()

    if config.bichar_pretrained_embed is True:
        p = Embed(path=config.bichar_pretrained_embed_file, words_dict=alphabet_static.word_alphabet.id2words,
                  embed_type=embed_types,
                  pad=paddingkey)
        bichar_pretrain_embed = p.get_embed()

    if config.char_pretrained_embed is True or config.bichar_pretrained_embed is True:
        embed_dict = {"char_pretrain_embed": char_pretrain_embed, "bichar_pretrain_embed": bichar_pretrain_embed}
        pcl.save(obj=embed_dict, path=os.path.join(config.pkl_directory, config.pkl_embed))

    return char_pretrain_embed, bichar_pretrain_embed


def load_model(config):
    """
    :param config:  config
    :return:  nn model
    """
    print("***************************************")
    model = JointPS(config)
    if config.use_cuda is True:
        model = model.cuda()
    if config.test is True:
        model = load_test_model(model, config)
    print(model)
    return model


def load_data(config):
    """
    :param config:  config
    :return: batch data iterator and alphabet
    """
    print("load data for process or pkl data.")
    train_iter, dev_iter, test_iter = None, None, None
    alphabet, alphabet_static = None, None
    if (config.train is True) and (config.process is True):
        print("process data")
        if os.path.exists(config.pkl_directory): shutil.rmtree(config.pkl_directory)
        if not os.path.isdir(config.pkl_directory): os.makedirs(config.pkl_directory)
        train_iter, dev_iter, test_iter, alphabet, alphabet_static = preprocessing(config)
        config.char_pretrain_embed, config.bichar_pretrain_embed = pre_embed(config=config, alphabet=alphabet,
                                                                             alphabet_static=alphabet_static)
    elif ((config.train is True) and (config.process is False)) or (config.test is True):
        print("load data from pkl file")
        # load alphabet from pkl
        alphabet_dict = pcl.load(path=os.path.join(config.pkl_directory, config.pkl_alphabet))
        print(alphabet_dict.keys())
        alphabet = alphabet_dict["alphabet"]
        alphabet_static = alphabet_dict["alphabet_static"]
        # load iter from pkl
        iter_dict = pcl.load(path=os.path.join(config.pkl_directory, config.pkl_iter))
        print(iter_dict.keys())
        train_iter, dev_iter, test_iter = iter_dict.values()
        # train_iter, dev_iter, test_iter = iter_dict["train_iter"], iter_dict["dev_iter"], iter_dict["test_iter"]
        # load embed from pkl
        if os.path.exists(os.path.join(config.pkl_directory, config.pkl_embed)):
            embed_dict = pcl.load(os.path.join(config.pkl_directory, config.pkl_embed))
            print(embed_dict.keys())
            embed = embed_dict["pretrain_embed"]
            config.pretrained_weight = embed

    return train_iter, dev_iter, test_iter, alphabet, alphabet_static



