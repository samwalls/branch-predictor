from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


# COMMON UTILITY FUNCTIONS


def import_data(path):
    data = pd.read_csv(path, sep='\s+', header=None, names=['target', 'taken'])
    data['taken'] = data['taken'] == 1
    return data


class SaturatingCounter(object):

    def __init__(self, taken=False, saturation=0):
        self.taken = taken
        self.saturation = saturation

    @property
    def taken(self):
        return self._taken

    @taken.setter
    def taken(self, value):
        self._taken = value

    @property
    def saturation(self):
        return self._saturation

    @saturation.setter
    def saturation(self, value):
        self._saturation = value


class Predictor(ABC):

    def __init__(self, data=None):
        if data is not None:
            self.data = data.copy()
        else:
            self.data = None

    @abstractmethod
    def predict_all(self):
        pass

    def analyse_total_accuracy(self):
        # total percentage accuracy
        total = len(self.data)
        correct = len(self.data[self.data['predict_correct'] == True])
        return correct / total

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value


class AlwaysPredictor(Predictor):

    def __init__(self, data=None, always_taken=False):
        super(AlwaysPredictor, self).__init__(data=data)
        self._always_taken = always_taken

    def predict_all(self):
        self.data['predict_taken'] = self._always_taken
        self.data['predict_correct'] = self.data['predict_taken'] == self.data['taken']
        return self.data


class SaturationPredictor(Predictor):

    def __init__(self, data=None, n_bits=1, table_size=512):
        super(SaturationPredictor, self).__init__(data=data)
        self._n_bits = n_bits
        if self._n_bits < 0:
            raise ValueError('cannot use n < 0 for n-bit saturation predictor')
        if self._n_bits == 0:
            self._max_saturation = 0
        else:
            self._max_saturation = 2 ** (n_bits - 1)
        self._table_size = table_size

    def predict_all(self):
        # generalises prediction from "strongly not taken" to "strongly taken" over 2 ^ (n_bits - 1) tolerance
        # n_bits = 1 produces a 1 bit predictor
        # n_bits = 2 produces a 2-bit predictor
        # etc..
        table = {}
        for i, row in self.data.iterrows():
            # make sure the address fits within the table (key counts higher than the table size will produce conflicts)
            addr = row['target'] % self._table_size
            if addr not in table.keys():
                table[addr] = SaturatingCounter()
            counter = table[addr]
            # provide the prediction value, evaluate after
            self.data.at[i, 'predict_taken'] = counter.taken
            # increase saturation value if misprediction
            if row['taken'] != counter.taken:
                counter.saturation = counter.saturation + 1
                self.data.at[i, 'predict_correct'] = False
            else:
                if counter.saturation > 0:
                    counter.saturation = counter.saturation - 1
                self.data.at[i, 'predict_correct'] = True
            # prediction is flipped after saturation by maximum mispredictions
            if counter.saturation >= self._max_saturation:
                counter.taken = not counter.taken
                counter.saturation = 0
        return self.data


class GSharePredictor(SaturationPredictor):

    def __init__(self, data=None, n_bits=1, table_size=512, addr_bits=8):
        super(GSharePredictor, self).__init__(data=data, n_bits=n_bits, table_size=table_size)
        self._addr_bits = addr_bits

    def predict_all(self):
        table = {}
        branch_history = 0
        for i, row in self.data.iterrows():
            # mask determines how much of the history/address to use
            mask = (2 ** self._addr_bits) - 1
            addr = row['target'] & mask
            # take XOR of target address and branch history
            branch_history = branch_history & mask
            addr = addr ^ branch_history
            # make sure the address fits within the table (key counts higher than the table size will produce conflicts)
            addr = addr % self._table_size
            if addr not in table.keys():
                table[addr] = SaturatingCounter()
            counter = table[addr]
            # provide the prediction value, evaluate after
            self.data.at[i, 'predict_taken'] = counter.taken
            # increase saturation value if misprediction
            if row['taken'] != counter.taken:
                counter.saturation = counter.saturation + 1
                self.data.at[i, 'predict_correct'] = False
            else:
                if counter.saturation > 0:
                    counter.saturation = counter.saturation - 1
                self.data.at[i, 'predict_correct'] = True
            # prediction is flipped after saturation by maximum mispredictions
            if counter.saturation >= self._max_saturation:
                counter.taken = not counter.taken
                counter.saturation = 0
            # shift a 1 or 0 to the end of the branch history depending on whether this row was taken
            branch_history = ((branch_history << 1) | (row['taken'] and 1 or 0))
        return self.data


class ProfilerPredictor(Predictor):

    def __init__(self, data=None, addr_bits=8):
        super(ProfilerPredictor, self).__init__(data=data)
        self._addr_bits = addr_bits
        self._probability = {}

    def _addr(self, row):
        # use subset of address to consolidate samples
        mask = (2 ** self._addr_bits) - 1
        return row['target'] & mask

    def train(self, _training_data):
        training_data = _training_data.copy()
        frequency = {}
        totals = {}
        probability = {}
        for i, row in training_data.iterrows():
            addr = self._addr(row)
            if addr not in frequency.keys():
                frequency[addr] = 0
                totals[addr] = 0
            totals[addr] = totals[addr] + 1
            # add to taken frequency if the branch was taken
            if row['taken']:
                frequency[addr] = frequency[addr] + 1
        for addr, freq in frequency.items():
            probability[addr] = frequency[addr] / totals[addr]
        self._probability = probability

    def predict_all(self):
        for i, row in self.data.iterrows():
            p = self._probability[self._addr(row)]
            prediction_n = np.random.choice(np.arange(0, 2), p=[1 - p, p])
            prediction = prediction_n == 1 and True or False
            self.data.at[i, 'predict_taken'] = prediction
            self.data.at[i, 'predict_correct'] = row['taken'] == prediction
        return self.data
