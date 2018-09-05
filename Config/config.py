
from configparser import ConfigParser
import os


class myconf(ConfigParser):
    """
     myconf
    """
    def __init__(self, defaults=None):
        ConfigParser.__init__(self, defaults=defaults)
        self.add_sec = "Additional"

    def optionxform(self, optionstr):
        """
        :param optionstr:
        :return:
        """
        return optionstr


class Configurable(myconf):
    """
        Configurable
    """
    def __init__(self, config_file):
        # config = ConfigParser()
        super().__init__()

        self.test = None
        self.train = None
        config = myconf()
        config.read(config_file)
        # if config.has_section(self.add_sec) is False:
        #     config.add_section(self.add_sec)
        self._config = config
        self.config_file = config_file

        print('Loaded config file Successfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, ":", v)
        if not os.path.isdir(self.save_direction):
            os.mkdir(self.save_direction)
        config.write(open(config_file, 'w'))

    def add_args(self, key, value):
        """
        :param key:
        :param value:
        :return:
        """
        self._config.set(self.add_sec, key, value)
        self._config.write(open(self.config_file, 'w'))

    # Embed
    @property
    def char_pretrained_embed(self):
        return self._config.getboolean('Embed', 'char_pretrained_embed')

    @property
    def bichar_pretrained_embed(self):
        return self._config.getboolean('Embed', 'bichar_pretrained_embed')

    @property
    def zeros(self):
        return self._config.getboolean('Embed', 'zeros')

    @property
    def avg(self):
        return self._config.getboolean('Embed', 'avg')

    @property
    def uniform(self):
        return self._config.getboolean('Embed', 'uniform')

    @property
    def nnembed(self):
        return self._config.getboolean('Embed', 'nnembed')

    @property
    def char_pretrained_embed_file(self):
        return self._config.get('Embed', 'char_pretrained_embed_file')

    @property
    def bichar_pretrained_embed_file(self):
        return self._config.get('Embed', 'bichar_pretrained_embed_file')

    # Data
    @property
    def train_file(self):
        return self._config.get('Data', 'train_file')

    @property
    def dev_file(self):
        return self._config.get('Data', 'dev_file')

    @property
    def test_file(self):
        return self._config.get('Data', 'test_file')

    @property
    def max_count(self):
        return self._config.getint('Data', 'max_count')

    @property
    def min_freq(self):
        return self._config.getint('Data', 'min_freq')

    @property
    def word_min_freq(self):
        return self._config.getint('Data', 'word_min_freq')

    @property
    def char_min_freq(self):
        return self._config.getint('Data', 'char_min_freq')

    @property
    def bichar_min_freq(self):
        return self._config.getint('Data', 'bichar_min_freq')

    @property
    def shuffle(self):
        return self._config.getboolean('Data', 'shuffle')

    @property
    def epochs_shuffle(self):
        return self._config.getboolean('Data', 'epochs_shuffle')

    # Save
    @property
    def save_pkl(self):
        return self._config.getboolean('Save', 'save_pkl')

    @property
    def pkl_directory(self):
        return self._config.get('Save', 'pkl_directory')

    @property
    def pkl_data(self):
        return self._config.get('Save', 'pkl_data')

    @property
    def pkl_alphabet(self):
        return self._config.get('Save', 'pkl_alphabet')

    @property
    def pkl_iter(self):
        return self._config.get('Save', 'pkl_iter')

    @property
    def pkl_embed(self):
        return self._config.get('Save', 'pkl_embed')

    @property
    def save_dict(self):
        return self._config.getboolean('Save', 'save_dict')

    @property
    def save_direction(self):
        return self._config.get('Save', 'save_direction')

    @property
    def dict_directory(self):
        return self._config.get('Save', 'dict_directory')

    @property
    def word_dict(self):
        return self._config.get('Save', 'word_dict')

    @property
    def label_dict(self):
        return self._config.get('Save', 'label_dict')

    @property
    def model_name(self):
        return self._config.get('Save', 'model_name')

    @property
    def save_best_model_dir(self):
        return self._config.get('Save', 'save_best_model_dir')

    @property
    def save_model(self):
        return self._config.getboolean('Save', 'save_model')

    @property
    def save_all_model(self):
        return self._config.getboolean('Save', 'save_all_model')

    @property
    def save_best_model(self):
        return self._config.getboolean('Save', 'save_best_model')

    @property
    def rm_model(self):
        return self._config.getboolean('Save', 'rm_model')

    # Model
    @property
    def model_bilstm(self):
        return self._config.getboolean("Model", "model_bilstm")

    @property
    def model_bilstm_context(self):
        return self._config.getboolean("Model", "model_bilstm_context")

    @property
    def lstm_layers(self):
        return self._config.getint("Model", "lstm_layers")

    @property
    def embed_dim(self):
        return self._config.getint("Model", "embed_dim")

    @property
    def embed_char_dim(self):
        return self._config.getint("Model", "embed_char_dim")

    @property
    def embed_bichar_dim(self):
        return self._config.getint("Model", "embed_bichar_dim")

    @property
    def rnn_hidden_dim(self):
        return self._config.getint("Model", "rnn_hidden_dim")

    @property
    def rnn_dim(self):
        return self._config.getint("Model", "rnn_dim")

    @property
    def pos_dim(self):
        return self._config.getint("Model", "pos_dim")

    @property
    def dropout_embed(self):
        return self._config.getfloat("Model", "dropout_embed")

    @property
    def dropout(self):
        return self._config.getfloat("Model", "dropout")

    @property
    def windows_size(self):
        return self._config.getint("Model", "windows_size")

    # Optimizer
    @property
    def adam(self):
        return self._config.getboolean("Optimizer", "adam")

    @property
    def sgd(self):
        return self._config.getboolean("Optimizer", "sgd")

    @property
    def learning_rate(self):
        return self._config.getfloat("Optimizer", "learning_rate")

    @property
    def weight_decay(self):
        return self._config.getfloat("Optimizer", "weight_decay")

    @property
    def clip_max_norm_use(self):
        return self._config.getboolean("Optimizer", "clip_max_norm_use")

    @property
    def clip_max_norm(self):
        return self._config.get("Optimizer", "clip_max_norm")

    @property
    def use_lr_decay(self):
        return self._config.getboolean("Optimizer", "use_lr_decay")

    @property
    def lr_rate_decay(self):
        return self._config.getfloat("Optimizer", "lr_rate_decay")

    @property
    def min_lrate(self):
        return self._config.getfloat("Optimizer", "min_lrate")

    @property
    def max_patience(self):
        return self._config.getint("Optimizer", "max_patience")

    # Train
    @property
    def num_threads(self):
        return self._config.getint("Train", "num_threads")

    @property
    def use_cuda(self):
        return self._config.getboolean("Train", "use_cuda")

    @property
    def set_use_cuda(self, use_cuda):
        self._config.set("Train", "use_cuda", use_cuda)

    @property
    def epochs(self):
        return self._config.getint("Train", "epochs")

    @property
    def early_max_patience(self):
        return self._config.getint("Train", "early_max_patience")

    @property
    def backward_batch_size(self):
        return self._config.getint("Train", "backward_batch_size")

    @property
    def batch_size(self):
        return self._config.getint("Train", "batch_size")

    @property
    def dev_batch_size(self):
        return self._config.getint("Train", "dev_batch_size")

    @property
    def test_batch_size(self):
        return self._config.getint("Train", "test_batch_size")

    @property
    def log_interval(self):
        return self._config.getint("Train", "log_interval")




