## Role of document in `DataUtils` directory ##
- DataUtils
	-  `Alphabet.py`  ------ Build vocab by train data or dev/test data

	- `Batch_Iterator.py` ------ Build batch and iterator for train/dev/test data, get train/dev/test iterator

	- `Common.py` ------ The file contains some common attribute, like random seeds, padding, unk and others

	- `eval.py` ------ The file is a eval script, For calculate F-score, recall, precision.

	- `Load_Pretrained_Embed.py`  ------ Loading Pre-trained word embedding( `glove` or `word2vec` ), now has two way: ose is `oov` (out of vocabulary) use zero word embedding, second is `oov` use average word embedding, will add another one is `oov` use random word embedding.

	- `Embed.py`  ------ overwrite `Load_Pretrained_Embed.py`, `zeros，avg, uniform, nn.Embedding for OOV`.

	-  `Embed_From_Pretrained.py` ------ `nn.Embedding()` from pre-trained word embedding, build a big vocabulary. It can use in `No-Finetune` word embedding.

	-  `Optim.py` ------ Encapsulate the `optimizer`.

	-  `Pickle.py` ------ Encapsulate the `pickle`.

	-  `utils.py` ------ common function.

	-  `state.py` ------ Decoder State

	-  `mainHelp.py` ------ main help file.
