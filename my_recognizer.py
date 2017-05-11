import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    for word_id, (X, lengths) in test_set.get_all_Xlengths().items():
        #import pdb; pdb.set_trace()
        word_prob = []
        for model_key, model_value in models.items():
            try:
                score = model_value.score(X, lengths)
                word_prob.append((score, model_key))
            except:
                word_prob.append((float("-inf"), model_key))
        probabilities.append(max(word_prob)[0])
        guesses.append(max(word_prob)[1])
    return probabilities, guesses
    
