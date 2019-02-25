## Config ##

- Use `ConfigParser` to config parameter  
	- `from configparser import ConfigParser`  .
	- Detail see `config.py` and `config.cfg`, please.  

- Following is `config.cfg` Parameter details.

- [Embed]  
	- `char_pretrained_embed` (True or False) ------ whether to use pretrained embedding.
	- `bichar_pretrained_embed` (True or False) ------ whether to use pretrained embedding.
	- `zeros`(True or False) ------ OOV by zeros .
	- `avg`(True or False) ------ OOV by avg .
	- `uniform`(True or False) ------ OOV by uniform .
	- `nnembed`(True or False) ------ OOV by nn.Embedding .
	- `char_pretrained_embed_file` --- pre train path
	- `bichar_pretrained_embed_file` --- pre train path

- [Data]  
	- `train-file/dev-file/test-file`(path)  ------ train/dev/test data path(`Data`).
	- `min_freq` /`word_min_freq`/`char_min_freq`/`bichar_min_freq`/(integer number) ------ The smallest Word frequency when build vocab.
	- `max_count ` (integer number) ------ The maximum instance for `debug`.
	- `shuffle/epochs-shuffle`(True or False) ------ shuffle data .

- [Save]
	- `pkl_directory` (path) ------ save pkl directory path.
	- `pkl_data` (path) ------ save pkl data path.
	- `pkl_alphabet` (path) ------ save pkl alphabet path.
	- `pkl_iter` (path) ------ save pkl batch iterator path.
	- `pkl_embed` (path) ------ save pkl pre-train embedding path.
	- `save_dict` (True or False) ------ save dict to file.
	- `dict_directory ` (path) ------ save dict directory path.
	- `word_dict ` (path) ------ save word dict path.
	- `label_dict ` (path) ------ save label dict path.
	- `save_model ` (True or False) ------ save model to file.
	- `save_all_model` (True or False) ------ save all model to file.
	- `save_best_model` (True or False) ------ save best model to file.
	- `model_name ` (str) ------ model name.
	- `rm_model` (True or False) ------ remove model to save space(now not use).

- [Model]
	- `model-bilstm` (True or False) ------ Bilstm model.
	- `model-***`(True or False) ------ *** model.
	- `lstm-layers` (integer) ------ number layers of lstm.
	- `embed-dim` (integer) ------ embedding dim = pre-trained embedding dim.
	- `embed_char_dim` (integer) ------ embedding dim = pre-trained embedding dim.
	- `embed_bichar_dim` (integer) ------ embedding dim = pre-trained embedding dim.
	- `rnn_hidden_dim` (integer) ------lstm hidden.
	- `rnn_dim ` (integer) ------ lstm input.
	- `pos_dim` (integer) ------pos dim.
	- `dropout-emb/dropout `(float) ------ dropout for prevent overfitting.
	- `windows-size` (integer) ------ Context window feature size.

- [Optimizer]
	- `adam` (True or False) ------ `torch.optim.Adam`
	- `sgd` (True or False)  ------ `torch.optim.SGD`
	- `learning-rate`(float) ------ learning rate(0.001, 0.01).
	- `weight-decay` (float) ------ L2.
	- `clip_max_norm_use` (True or False) ------ use util.clip.
	- `clip-max-norm` (Integer number) ------ clip-max-norm value.
	- `use_lr_decay` (True or False) ------ use lr decay.
	- `lr_rate_decay`(float) ------ lr decay value.
	- `min_lrate `(float) ------ minimum lr.
	- `max_patience `(Integer) ------ patience to decay.

- [Train]
	- `num-threads` (Integer) ------ threads.
	- `epochs` (Integer) ------ maximum train epochs
	- `early_max_patience` (Integer) ------ maximum dev no improvment times for early stop.
 	- `backward_batch_size` (Integer) ------ multiple-batch to update parameters.
	- `batch-size/dev-batch-size/test-batch-size` (Integer) ------ number of batch
	- `log-interval`(Integer) ------ steps of print log.