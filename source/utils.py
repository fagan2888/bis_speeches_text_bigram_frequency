import pickle
from datetime import datetime, timedelta
import os
import re
import csv
import time
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import stopwords
from BigramCollocationFinder_custom import BigramCollocationFinder
from sklearn.feature_extraction.text import CountVectorizer


def create_dirs(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        current_pkl = pickle.load(f)
    print('Completed loading:', filepath)
    return current_pkl


def end_pkl(target_to_save, pkl_path, start=None):
    with open(pkl_path, 'wb') as f:
        pickle.dump(target_to_save, f)
    print('Creating .pkl completed: ', pkl_path)

    if start is not None:
        elapsed_time = time.time() - start
        elapsed_time_format = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print('END. Elapsed time: ', elapsed_time_format)


def get_now_time_str():
    return datetime.now().strftime("%Y%m%d-%H-%M-%S")


def get_str_concat(*args):
    _str = ""
    firstLine = True
    for idx, arg in enumerate(args):
        if idx == 0:
            _str += arg
            firstLine = False
            continue
        _str = _str + "_" + arg
    return _str


def dates_between(start_dt, end_dt):
    dates = []
    cursor = start_dt
    while cursor <= end_dt:
        if cursor.strftime('%Y-%m-%d') not in dates:
            dates.append(cursor.strftime('%Y-%m-%d'))
        cursor += timedelta(days=1)
    return dates


def get_target_list_filtered_by_author_and_years(df, target_author, target_dates):
    df_author = df[df['short_info'].str.contains(target_author, na=False)]
    df_dates = df_author[df_author.date.isin(target_dates)]
    
    target_list = list()
    for _, _target_row in df_dates.iterrows():
        target_list.append(_target_row.to_dict())
        
    return target_list


number_pattern = re.compile('\d+')
alphabet_regex = re.compile('[^a-zA-Z]')
doublespace_pattern = re.compile('\s+')


def get_sentences(content, remove_number=False, lower_case=False, alphabet_only=False):
    if remove_number:
        content = number_pattern.sub(' ', content)
    if lower_case:
        content = content.lower()

    sentences = sent_tokenize(content)
    for i in range(len(sentences)):
        if alphabet_only:
            sentences[i] = alphabet_regex.sub(' ', sentences[i])
            sentences[i] = doublespace_pattern.sub(' ', sentences[i])
        sentences[i] = sentences[i].strip()
    return sentences


noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
adjective_tags = ['JJ', 'JJR', 'JJS']
adverb_tags = ['RB', 'RBR', 'RBS']
verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def get_pos_tagged_words(content, lower_case=False, lemmatize=False, stemming=False, alphabet_only=False,
                         remove_len_one=False, include_noun=True, include_adjective=True, include_adverb=True,
                         include_verb=True):
    if lower_case:
        content = content.lower()
    words = word_tokenize(content)

    lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()
    for i in range(len(words)):
        if lemmatize:
            words[i] = lemmatizer.lemmatize(words[i])
        if stemming:
            words[i] = porter_stemmer.stem(words[i])
        if alphabet_only:
            words[i] = alphabet_regex.sub(' ', words[i])
            words[i] = doublespace_pattern.sub(' ', words[i])
        words[i] = words[i].strip()
    words = [word for word in words if word != '']

    pos_tagged_words = pos_tag(words)
    if remove_len_one:
        pos_tagged_words = [(x, y) for (x, y) in pos_tagged_words if len(x) != 1]

    tag_list = list()
    if include_noun:
        tag_list.extend(noun_tags)
    if include_adjective:
        tag_list.extend(adjective_tags)
    if include_adverb:
        tag_list.extend(adverb_tags)
    if include_verb:
        tag_list.extend(verb_tags)
    if len(tag_list) != 0:
        pos_tagged_words = [(x, y) for (x, y) in pos_tagged_words if y in tag_list]
    else:
        pos_tagged_words = [(x, y) for (x, y) in pos_tagged_words if y not in ('CC', 'IN', 'DT', 'TO')]

    return pos_tagged_words


def _period_dict(target_dict, _period, start, end):
    for _month in [str('%02d' % i) for i in range(start, end + 1)]:
        target_dict[_month] = _period
    return target_dict

def monthly_dict():
    target_dict = dict()
    for i in range(1, 12+1):
        _period = str(i)
        start, end = i, i
        target_dict = _period_dict(target_dict, _period, start, end)
    return target_dict

def quarterly_dict():
    target_dict = dict()
    _period, start, end = 'Q1', 1, 3
    target_dict = _period_dict(target_dict, _period, start, end)
    _period, start, end = 'Q2', 4, 6
    target_dict = _period_dict(target_dict, _period, start, end)
    _period, start, end = 'Q3', 7, 9
    target_dict = _period_dict(target_dict, _period, start, end)
    _period, start, end = 'Q4', 10, 12
    target_dict = _period_dict(target_dict, _period, start, end)
    return target_dict


def semiannually_dict():
    target_dict = dict()
    _period, start, end = '1H', 1, 6
    _period_dict(target_dict, _period, start, end)
    _period, start, end = '2H', 7, 12
    _period_dict(target_dict, _period, start, end)
    return target_dict


def annually_dict():
    target_dict = dict()
    _period, start, end = '', 1, 12
    _period_dict(target_dict, _period, start, end)
    return target_dict


def dict_val_as_list_append(target_dict, index_key, val):
    if index_key not in target_dict:
        target_dict[index_key] = list()
    target_dict[index_key].append(val)
    return target_dict


def get_grouped_list_in_dict(target_list, period='quarterly'):  # 'quarterly', 'semiannually', 'annually'
    if period == 'monthly':
        period_dict = monthly_dict()
    elif period == 'quarterly':
        period_dict = quarterly_dict()
    elif period == 'semiannually':
        period_dict = semiannually_dict()
    elif period == 'annually':
        period_dict = annually_dict()

    result_dict = dict()
    for _item in target_list:
        _date = _item['date']
        _year = _date[:4]
        _month = _date[5:7]
        grouping_key = get_str_concat(_year, period_dict[_month])
        result_dict = dict_val_as_list_append(result_dict, grouping_key, _item)
    return result_dict


def stopwords_set_filter(nltk_stopwords='english', custom_list=None):
    stopset = set()
    if nltk_stopwords is not None:
        stopset.update(stopwords.words(nltk_stopwords))
    if custom_list is not None:
        for _custom in custom_list:
            stopset.add(_custom)
    filter_stops = lambda w: w in stopset
    return stopset, filter_stops


def words_from_pos_tagged_words(pos_tagged_words):
    return [x.strip() for (x, y) in pos_tagged_words]


def get_one_dimensional_words(doc_list):
    words = list()
    for doc in doc_list:
        for unigrams_with_pos in doc['unigrams_by_sentence']:
            words.extend(words_from_pos_tagged_words(unigrams_with_pos))
    return words


def bigram_freq_rank_dict(finder, bigram_max_rank=None):
    bigram_dict = dict()
    _rank = 0
    for _bigram, _freq in sorted(finder.ngram_fd.items(), key=lambda t: t[-1], reverse=True):  # frequency descending
        _rank += 1
        if _bigram[0] == _bigram[1]:
            _rank -= 1
            continue
        bigram_dict[_bigram[0] + '-' + _bigram[1]] = (_freq, _rank)
        if bigram_max_rank is not None and _rank == bigram_max_rank:
            break
    return bigram_dict


def bigram_collocation_finder_custom(lookup_finder, word_list, bigram_window_size=2, filter_stops=None):
    finder = BigramCollocationFinder.from_words(lookup_finder, word_list, window_size=bigram_window_size)
    if filter_stops is not None:
        finder.apply_word_filter(filter_stops)
    return finder


def bow_unigram_freq_dict(word_list, stopset=None):
    vectorizer = CountVectorizer(stop_words=stopset)
    X = vectorizer.fit_transform(word_list)
    terms = vectorizer.get_feature_names()
    freqs = X.sum(axis=0).A1
    unigram_bow_dict = dict(zip(terms, freqs))
    return unigram_bow_dict


def get_0_if_None(content):
    if content is None:
        return 0
    return content


def words_from_range(_period_dict, range_start, range_end):
    _target_periods = [j for j in range(range_start, range_end)]
    _target_period_list_of_list_of_doc = [_period_dict[_period_j] for j, _period_j in
                                          enumerate(sorted(_period_dict.keys())) if j in _target_periods]
    _target_doc_list = list()
    for _target_period_list_of_doc in _target_period_list_of_list_of_doc:
        for doc in _target_period_list_of_doc:
            _target_doc_list.append(doc)
    _target_words = get_one_dimensional_words(_target_doc_list)
    return _target_words


def start_csv(csv_filepath, csv_delimiter=','):
    f = open(csv_filepath, 'w', encoding='utf-8-sig', newline='')
    wr = csv.writer(f, delimiter=csv_delimiter)
    return f, wr


def end_csv(f, csv_filepath):
    f.close()
    print('Creating .csv file completed: ', csv_filepath)

def get_index_dict(fred_gdp_index_dict_pkl_filepath, fred_index_filepath_dict):
    if os.path.exists(fred_gdp_index_dict_pkl_filepath):
        index_dict = load_pkl(fred_gdp_index_dict_pkl_filepath)
    else:
        index_dict = dict()

        for period_category, index_data_filepath in fred_index_filepath_dict.items():
            index_dict[period_category] = dict()

            f = open(index_data_filepath, 'r', encoding='utf-8')
            rdr = csv.reader(f)
            firstline = True
            for line in rdr:
                if firstline:
                    firstline = False
                    continue
                _period = line[0]
                if _period == '':
                    continue
                _val = line[1]
                index_dict[period_category][_period] = _val
            f.close()

        end_pkl(index_dict, fred_gdp_index_dict_pkl_filepath)
    return index_dict

def rescale_(values, scaler):
    values = values.reshape((len(values), 1))
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)
    return normalized