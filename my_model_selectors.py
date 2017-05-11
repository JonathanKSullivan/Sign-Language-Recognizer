import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def score(self, num_states):
        model = self.base_model(n)
        features = len(self.X)
        logL = model.score(self.X, self.lengths)
        logN = np.log(features)
        parameters = num_states ** 2 + 2 * num_states * n - 1
        d = model.n_features
        p = num_states ** 2 + 2 * d * num_states - 1
        return -2.0 * logL + p * logN, model

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection using CV
        try:
            bics = []
            for num_states in range(self.min_n_components, self.max_n_components + 1):
                bics.append(score(num_states))
            best_model = max(bics)[1]
            return best_model
        except:
            return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def score(self):
        model = self.base_model(n)
        this_log = model.score(self.X, self.lengths) 
        rest_logs = [model.score(X, lengths) for word, (X, lengths) in self.hwords.items() if word != self.this_wor]
        return this_log - np.mean(rest_logs), model

    def select(self):
        try:
            dics = []
            for num_states in range(self.min_n_components, self.max_n_components + 1):
                dics.append(score(num_states))
            best_model = max(dics)[1]
            return best_model
        except:
            return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def score(self, num_states):
        logLikelihoods = []
        split_method = KFold(n_splits=2)
        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
            model = self.base_model(num_states)
            X, lengths = combine_sequences(cv_test_idx, self.sequences)
            logLikelihoods.append(model.score(X, lengths))
        self.X, self.lengths = self.hwords[self.this_word]
        return np.mean(logLikelihoods), model

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection using CV
        try:
            avg_log_likelihood = []
            for num_states in range(self.min_n_components, self.max_n_components + 1):
                avg_log_likelihood.append(self.score(num_states))
            best_model = max(avg_log_likelihood)[1]
            return best_model
        except:
            return self.base_model(self.n_constant)
        

