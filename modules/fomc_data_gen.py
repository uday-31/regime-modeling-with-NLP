import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense

FOMC_STOP_WORDS = ["federal", "reserve", "board", "meeting", "committee"]
def custom_stop_words():
    stop_words = list(ENGLISH_STOP_WORDS) + FOMC_STOP_WORDS
    return stop_words

def join_tokens(tokens):
    return " ".join(tokens)


def remove_stop_words(tokens):
    stop_words = custom_stop_words()
    tokens = [word.lower() for word in tokens if word.isalpha()]
    result = [w.lower() for w in tokens if w.lower() not in stop_words]
    return result

def clean_line(line):
  if '"' in line:
    l_idx = line.index('"')
    r_idx = line.rindex('"') + 1
    res = line[:l_idx - 1].split(",") + [line[l_idx + 1:r_idx - 1]] + [line[r_idx + 1:].strip()]
  else:
    res = line.split(",")
  return res

def truncate_text_by_token(text: str):
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

    return text.split(split_by)[1]

def generate_fomc_data():
    try:
      df = pd.read_csv("fomc_documents.csv")
    except Exception as e:
      with open('fomc_documents.csv', mode='r') as fin:
        lines = fin.readlines()
      cols = lines[0].strip().split(",")
      lines = list(map(clean_line, lines[1:]))
      assert (np.array(list(map(len, lines))) != 5).sum() == 0
      df = pd.DataFrame(lines)
      df.columns = cols
    df = df[df.document_kind.isin(["historical_minutes", "minutes", "minutes_of_actions"])]
    df['meeting_year'] = df.meeting_date.dt.year
    df = df[df.meeting_year >= 1985]
    df = df.text.apply(truncate_text_by_token)
    return df

def tokenize_data(df):
    df['tokens'] = df.text.apply(nltk.word_tokenize)
    df["tokens_wo_stop"] = df.tokens.apply(remove_stop_words)
    df["new_text"] = df.tokens_wo_stop.apply(join_tokens)

def generate_gensim_input(df, dict_gensim_statements=None):
  gensim_statements = df["tokens_wo_stop"] .tolist()
  if dict_gensim_statements is None: 
    dict_gensim_statements = Dictionary(gensim_statements)
  bow_gensim_statements = [dict_gensim_statements.doc2bow(d) for d in gensim_statements]
  return gensim_statements, dict_gensim_statements, bow_gensim_statements

def convert_to_dense_tfidf_matrix(stmts, stmts_dict, bow, tfidf_model=None):
  if tfidf_model is None: 
    tfidf_model = TfidfModel(bow)
  tfidf_statements = tfidf_model[bow]  
  num_docs = len(stmts)
  num_terms = len(stmts_dict.keys())
  corpus_tfidf_dense = corpus2dense(tfidf_statements, num_terms, num_docs)
  return tfidf_model, corpus_tfidf_dense.T

   