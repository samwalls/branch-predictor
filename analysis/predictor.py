from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# COMMON UTILITY FUNCTIONS

def import_data(path):
    data = pd.read_csv(path, sep='\s+', header=None, names=['target', 'taken'])
    data['taken'] = data['taken'] == 1
    return data


# MEASUREMENT AND TESTING


def run_all(test_set):
    results = pd.DataFrame()
    accuracy = {}
    for name, predictor in test_set.items():
        print("running prediction for", name)
        results[name] = predictor.predict_all()['predict_correct']
        accuracy[name] = predictor.analyse_total_accuracy()
    accuracy_table = pd.DataFrame(data=list(accuracy.items()), columns=['name', 'accuracy'])
    accuracy_table.set_index('name')
    return results, accuracy_table


def run_plot_compare(test_set=None, name=None, rolling_window=1):
    results, accuracy = run_all(test_set)

    plt.rcParams.update({'font.size': 8})

    fig1, ax1 = plt.subplots(dpi=100)
    accuracy.plot(
        kind='bar',
        ylim=(0.0, 1.0),
        title='Average Comparison' + (name is not None and ': ' + name or ''),
        ax=ax1,
        y='accuracy',
        legend=False,
        figsize=(3, 3)
    )
    ax1.set_xlabel('method')
    ax1.set_ylabel('total accuracy')

    fig2, ax2 = plt.subplots(dpi=150)
    results.astype(float).rolling(rolling_window).mean().plot(
        linewidth=1,
        ylim=(0.0, 1.0),
        title='Instantaneous Comparison' + (name is not None and ': ' + name or ''),
        ax=ax2,
        figsize=(8, 5)
    )
    ax2.set_xlabel('branch number')
    ax2.set_ylabel('rolling mean correct predictions')

    return results, accuracy, fig1, fig2

# BRANCH PREDICTOR CLASSES


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
        if data is not None:
            self._base_addr = data['target'].min()
        else:
            self._base_addr = 0
        self._frequency = {}
        self._totals = {}
        self._probability = {}
        self.clear()

    def clear(self):
        self._frequency = {}
        self._totals = {}
        self._probability = {}

    def _addr(self, row, base_addr=None):
        # use subset of address to consolidate samples
        mask = (2 ** self._addr_bits) - 1
        # make address relative to base address
        # (the intuition for this is to generalise one training run across to other runs)
        return (row['target'] - (base_addr is None and self._base_addr or base_addr)) & mask

    def train(self, _training_data):
        training_data = _training_data.copy()
        for i, row in training_data.iterrows():
            addr = self._addr(row)
            if addr not in self._frequency.keys():
                self._frequency[addr] = 0
                self._totals[addr] = 0
            self._totals[addr] = self._totals[addr] + 1
            # add to taken frequency if the branch was taken
            if row['taken']:
                self._frequency[addr] = self._frequency[addr] + 1
        for addr, freq in self._frequency.items():
            self._probability[addr] = self._frequency[addr] / self._totals[addr]
        return self

    def predict_all(self):
        for i, row in self.data.iterrows():
            p = self._probability[self._addr(row)]
            prediction_n = np.random.choice(np.arange(0, 2), p=[1 - p, p])
            prediction = prediction_n == 1 and True or False
            self.data.at[i, 'predict_taken'] = prediction
            self.data.at[i, 'predict_correct'] = row['taken'] == prediction
        return self.data


class NgramProfilerPredictor(ProfilerPredictor):

    def __init__(self, data=None, addr_bits=8, n=1, default_take_probability=0.5):
        # n = 1 by default, meaning that by default this will perform the same as the base ProfilerPredictor
        super(NgramProfilerPredictor, self).__init__(data=data, addr_bits=addr_bits)
        self._n = n
        # the probability of a taken branch if there exists no key in the probability table
        self._default_take_probability = default_take_probability

    def train(self, _training_data):
        training_data = _training_data.copy()
        base_addr = training_data['target'].min()
        # create empty tuple of n elements
        ngram = ((),) * self._n
        for i, row in training_data.iterrows():
            ngram = ngram + (self._addr(row), )
            # take a tuple of the last n elements after adding an element
            ngram = ngram[-self._n:]
            if ngram not in self._frequency.keys():
                self._frequency[ngram] = 0
                self._totals[ngram] = 0
            self._totals[ngram] = self._totals[ngram] + 1
            # add to taken frequency if the branch was taken
            if row['taken']:
                self._frequency[ngram] = self._frequency[ngram] + 1
        for addr, freq in self._frequency.items():
            self._probability[addr] = self._frequency[addr] / self._totals[addr]
        return self

    def predict_all(self):
        ngram = ((),) * self._n
        for i, row in self.data.iterrows():
            ngram = ngram + (self._addr(row),)
            # take a tuple of the last n elements after adding an element
            ngram = ngram[-self._n:]
            # get probability of this ngram resulting in a taken branch
            if ngram in self._probability.keys():
                p = self._probability[ngram]
            else:
                # TODO this value is arbitrary, what else could be done?
                p = self._default_take_probability
            prediction_n = np.random.choice(np.arange(0, 2), p=[1 - p, p])
            prediction = prediction_n == 1 and True or False
            self.data.at[i, 'predict_taken'] = prediction
            self.data.at[i, 'predict_correct'] = row['taken'] == prediction
        return self.data
