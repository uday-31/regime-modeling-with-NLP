import numpy as np
import pandas as pd
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


def remove_names_from_minutes(text: str):
    """
    This function removes all names from the start of FED Minutes by relying on
    the fact that the phrases 'the manager' and 'unanimous' tend to appear at
    the end of the initial string of names.

    @param tet(str): text which needs to have names removed from the start
    @returns res(str): portion of text after first occurence of 'the manager'
                       or 'unanimous'
    """
    text = text.lower()
    split_by = ''
    if 'the manager' in text and 'unanimous' in text:
        if text.index('the manager') > text.index('unanimous'):
            split_by = 'unanimous'
        else:
            split_by = 'the manager'
    elif 'the manager' in text:
        split_by = 'the manager'
    elif 'unanimous' in text:
        split_by = 'unanimous'
    else:
        raise ValueError('Neither in text!')
    
    res = text.split(split_by)[1]
    return res


def tokenizer_wo_stopwords(text: str):
    """
    This function prepares raw text by tokenizing it and removing all stop
    words (based on nltk stopwords).

    @param text(str): raw text which needs to be prepared for analysis
    @returns res(str): string representation of text without stopwords
    """
    tokens = nltk.word_tokenize(text)
    words = [word.lower() for word in tokens if word.isalpha()]
    words_wo_stop = [w.lower() for w in words if w.lower() not in ENGLISH_STOP_WORDS]
    res = ' '.join(words_wo_stop)
    return res


def manual_tf_idf(text: pd.Series):
    """
    This function manually computes the TF-IDF values for the given column of
    documents, in order to avoid the incorrect computations performed by
    sklearn's native implementation.

    @param text(pd.Series): the (n, 1) column of n documents for which TF-IDF
                            is being computed
    @returns tf_idf_df(np.ndarray): the (n, n_words) representing TF-IDF values
    """
    # Get number of documents
    n_docs = text.shape[0]

    # Generate bag-of-words matrix
    def_vectorizer = CountVectorizer(token_pattern='[a-zA-Z]+')
    word_bow_matrix = def_vectorizer.fit_transform(text)
    word_bow_df = pd.DataFrame(
        word_bow_matrix.toarray(),
        columns=def_vectorizer.get_feature_names_out()
    )

    # Create TF matrix
    tf_df = word_bow_df / word_bow_df.sum(axis=1).values.reshape(n_docs, 1)

    # Compute IDF values
    idf = np.log(n_docs / (word_bow_df / word_bow_df.values).sum(axis=0))

    # Manually create TF-IDF matrix
    tfidf_df = tf_df * idf

    return tfidf_df


if __name__ == "__main__":
    print(f"Please import this module as a library!")
