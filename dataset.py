import random
import torch
import numpy
import csv

class Loader:

    def __init__(self, train_file, valid_file, test_file):
        self.train = Loader.load(train_file)
        self.valid = Loader.load(valid_file)
        self.test = Loader.load(test_file)

    @staticmethod
    def load(filename):
        result = []
        with open(filename, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            next(spamreader, None)  # skip the headers
            for row in spamreader:
                res = {}
                res['question'] = str(row[0])
                res['answer'] = str(row[1])
                res['label'] = int(row[2])
                result.append(res)
        return result

    def get(self):
        return self.train, self.valid, self.test


class DatasetManager:

    def __init__(self, train_file, valid_file, test_file,
                 max_length,
                 tokenizer,
                 pad_position='right',
                 device='cpu',
                 #batch_size=32,
                 logger=None,
                 mask_padding_with_zero=True,
                 pad_token_segment_id=0,
                 pad_token=0
    ):

        self.device = device
        #self.batch_size = batch_size

        # padding options
        self.pad_token_segment_id = pad_token_segment_id
        self.pad_position = pad_position
        self.pad_token = pad_token

        # max answer and question length
        self.max_length = max_length

        # dataset mode
        #self.mode = None

        # sentences tokenizer
        self.tokenizer = tokenizer

        # logger
        self.logger = logger

        # pad real tokens with 1 and pads with 0
        self.mask_padding_with_zero = mask_padding_with_zero

        # files
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        
        # get parsed files
        self.train, self.valid, self.test = Loader(self.train_file, self.valid_file, self.test_file).get()

        # get statistics before padding and tokenization
        self.set_statistics()

        # tokenize and pad
        self.train = self.tokenize(self.train, self.max_length)
        self.valid = self.tokenize(self.valid, self.max_length)
        self.test = self.tokenize(self.test, self.max_length)


    @staticmethod
    def padding(sequence, max_length, pad_position='right', pad_token=0):
        """
        Pad sequence at the given max_length length.
        pad_token specifies which token should be used to pad.
        pad_position can be 'right' or 'left'.
        """
        padding_length = max_length - len(sequence)
        if pad_position == 'right':
            return sequence + [pad_token] * padding_length
        else:
            return [pad_token] * padding_length + sequence

    def tokenize(self, data, max_length):
        res = []
        for triple in data:
            inputs = self.tokenizer.encode_plus(
                triple['question'],
                triple['answer'],
                add_special_tokens=True,
                max_length=max_length,
            )
            #print(inputs, self.tokenizer.decode(inputs['input_ids'])); exit()

            if self.logger is not None and 'num_truncated_tokens' in inputs and inputs['num_truncated_tokens'] > 0:
                self.logger.info('Attention! you are cropping tokens (swag task is ok). '
                        'If you are training ARC and RACE and you are poping question + options,'
                        'you need to try to use a bigger max seq length!')

            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if self.mask_padding_with_zero else 0] * len(input_ids)

            input_ids = DatasetManager.padding(
                input_ids,
                max_length,
                pad_position=self.pad_position,
                pad_token=self.pad_token
            )
            #print(self.tokenizer.decode(input_ids)); exit()
            attention_mask = DatasetManager.padding(
                attention_mask,
                max_length,
                pad_position=self.pad_position,
                pad_token=(0 if self.mask_padding_with_zero else 1)
            )
            token_type_ids = DatasetManager.padding(
                token_type_ids,
                max_length,
                pad_position=self.pad_position,
                pad_token=self.pad_token_segment_id
            )
            label = triple['label']

            # assert to have done a good job
            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length

            res.append(
                (input_ids, attention_mask, token_type_ids, label)
            )

        # transform to torch tensor with int type (are IDs)
        all_input_ids = torch.tensor([x[0] for x in res], dtype=torch.int64, device=self.device)
        all_attention_mask = torch.tensor([x[1] for x in res], dtype=torch.int64, device=self.device)
        all_token_type_ids = torch.tensor([x[2] for x in res], dtype=torch.int64, device=self.device)
        all_labels = torch.tensor([x[3] for x in res], dtype=torch.int64, device=self.device)

        #print(all_input_ids[:10]); exit()

        return torch.utils.data.TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    """
    def _train(self):
        self.mode = 'training'

    def _valid(self):
        self.mode = 'validation'

    def _test(self):
        self.mode = 'testing'
        
    def _init(self):
        if self.mode is None:
            raise EOFError('You should first set a mode with instance._train(), instance._valid() or instance._test()')
        self.index = 0
        if self.mode == 'training':
            random.shuffle(self.train)
        elif self.mode == 'validation':
            random.shuffle(self.valid)
        elif self.mode == 'testing':
            random.shuffle(self.test)
        else:
            raise ValueError('Inconsistecy error, someout has changes self.mode to a non-valid value')
    
    def _batches(self, data):
        batches = []
        assert len(data[0]) == len(data[1]) == len(data[2]) == len(data[3]), \
            "Dataset error, number of sample, masks, tokens or labels not matching"
        n_samples = data[0]
        for index in range(len(n_samples) // self.batch_size + 1):
            start_index = index * self.batch_size
            end_index = min((index + 1) * self.batch_size, len(data))
            
            batches.append(
                tuple(d[start_index: end_index] for d in data)    
            )
        return batches
        
    def batches(self):
        if self.mode is None:
            raise EOFError('You should first set a mode with instance._train(), instance._valid() or instance._test()')
        if self.mode == 'training':
            return self._batches(self.train)
        elif self.mode == 'validation':
            return self._batches(self.valid)
        elif self.mode == 'testing':
            return self._batches(self.test)
        else:
            raise ValueError('Inconsistecy error, someout has changes self.mode to a non-valid value')
    """
    
    # to get the statistics
    def get_statistics(self):
        return self.statistics

    # to set the statistics
    def set_statistics(self):
        self.statistics = (
            numpy.mean(list(map(lambda a: len(a['question']), self.train + self.valid + self.test))), # average question len
            numpy.mean(list(map(lambda a: len(a['answer']), self.train + self.valid + self.test))), # average answer len
            len(list(filter(lambda a: a['label'] == 1, self.train + self.valid + self.test))), # number of correct answers
            len(list(filter(lambda a: a['answer'] == 0, self.train + self.valid + self.test))) # number of wrong answers
        )
