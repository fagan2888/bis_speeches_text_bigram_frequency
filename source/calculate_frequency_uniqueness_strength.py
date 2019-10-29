from config import parameters
from utils import *
import numpy as np
import math
import time
import os

bigram_window_size = parameters.bigram_window_size
bigram_max_rank = parameters.bigram_max_rank
stopword_list = parameters.stopword_list
ws_quarterly = parameters.ws_quarterly
ws_semiannually = parameters.ws_semiannually
ws_annually = parameters.ws_annually
strength_alpha = parameters.strength_alpha
output_base_dir = parameters.output_base_dir

quarterly_filepath = 'target-list-units-grouped_quarterly_20191028-21-52-08.pkl'
semiannually_filepath = 'target-list-units-grouped_semiannually_20191028-21-52-09.pkl'
annually_filepath = 'target-list-units-grouped_annually_20191028-21-52-09.pkl'

bigram_by_period_dict_of_list_filepath = os.path.join(output_base_dir, 'bigram_by_period_dict_of_list.pkl')
bigram_uniqueness_strength_pkl_filepath = os.path.join(output_base_dir,
                                                       get_str_concat('bigram-uniqueness-strength',
                                                                      get_now_time_str()) + '.pkl')
bigram_uniqueness_strength_csv_filepath = os.path.join(output_base_dir,
                                                       get_str_concat('bigram-uniqueness-strength',
                                                                      get_now_time_str()) + '.csv')

stopset, filter_stops = stopwords_set_filter('english', stopword_list)

quarterly_target_doc_dict = load_pkl(os.path.join(parameters.output_base_dir, quarterly_filepath))
semiannually_target_doc_dict = load_pkl(os.path.join(parameters.output_base_dir, semiannually_filepath))
annually_target_doc_dict = load_pkl(os.path.join(parameters.output_base_dir, annually_filepath))

period_dict = {'quarterly': (quarterly_target_doc_dict, ws_quarterly),
               'semiannually': (semiannually_target_doc_dict, ws_semiannually),
               'annually': (annually_target_doc_dict, ws_annually)}


def get_bigram_by_period_dict_of_list():
    if os.path.exists(bigram_by_period_dict_of_list_filepath):
        bigram_by_period_dict_of_list = load_pkl(bigram_by_period_dict_of_list_filepath)
    else:
        print('=' * 5, '\n Start creating bigram_by_period_dict_of_list')
        start = time.time()
        bigram_by_period_dict_of_list = {'quarterly': list(), 'semiannually': list(), 'annually': list()}
        for _period_category, (_period_dict, _) in period_dict.items():
            for i, _period in enumerate(sorted(_period_dict.keys())):
                _this_words = words_from_range(_period_dict, i, i + 1)
                _this_finder = nltk_bigram_collocation_finder(_this_words, bigram_window_size, filter_stops)
                _this_bigram_freq_rank_dict = bigram_freq_rank_dict(_this_finder, bigram_max_rank)

                bigram_by_period_dict_of_list[_period_category].append(_this_bigram_freq_rank_dict)
        end_pkl(bigram_by_period_dict_of_list, bigram_by_period_dict_of_list_filepath, start)
    return bigram_by_period_dict_of_list


def main():
    start = time.time()
    bigram_by_period_dict_of_list = get_bigram_by_period_dict_of_list()
    final_dict = {'quarterly': dict(), 'semiannually': dict(), 'annually': dict()}
    f, wr = start_csv(bigram_uniqueness_strength_csv_filepath, csv_delimiter=',')
    for _period_category, (_period_dict, _ws) in period_dict.items():
        wr.writerow([_period_category, 'bigram', 'raw_frequency', 'uniqueness', 'strength'])

        for k, _period in enumerate(sorted(_period_dict.keys())):
            if k - _ws <= 0:
                continue

            print('Processing', _period)
            # Uniqueness: Current Bigram frequency
            _current_words = words_from_range(_period_dict, k - _ws, k + 1)
            _current_finder = nltk_bigram_collocation_finder(_current_words, bigram_window_size, filter_stops)
            _current_bigram_freq_rank_dict = bigram_freq_rank_dict(_current_finder, bigram_max_rank)

            # Uniqueness: Reference Bigram frequency
            _reference_words = words_from_range(_period_dict, 0, k - _ws)
            _reference_finder = nltk_bigram_collocation_finder(_reference_words, bigram_window_size, filter_stops)
            _reference_bigram_freq_rank_dict = bigram_freq_rank_dict(_reference_finder, bigram_max_rank)

            final_dict[_period_category][_period] = dict()
            for _bigram, (_freq, _rank) in sorted(_current_bigram_freq_rank_dict.items(),
                                                  key=lambda t: t[-1][1]):  # ranking ascending
                # Uniqueness: score
                _numerator = _freq
                if _reference_bigram_freq_rank_dict.get(_bigram) is None:
                    _reference_freq = 0
                else:
                    _reference_freq = _reference_bigram_freq_rank_dict.get(_bigram)[0]
                _denominator = _reference_freq + 1
                _uniqueness = np.log(_numerator / _denominator)

                # Strength: score
                _strength = 0
                for i_ in range(k - _ws, k + 1):
                    _that_bigram_dict = bigram_by_period_dict_of_list[_period_category][i_]
                    if _that_bigram_dict.get(_bigram) is None:
                        _that_freq, _that_rank = 0, 0
                    else:
                        _that_freq, _that_rank = _that_bigram_dict.get(_bigram)

                    _first_term = _that_freq / (_that_rank + 1)
                    _second_term = math.pow(strength_alpha, -i_ + k)
                    _strength += _first_term * _second_term

                final_dict[_period_category][_period][_bigram] = (_freq, _uniqueness, _strength)
                wr.writerow([_period, _bigram, _freq, _uniqueness, _strength])
    end_csv(f)
    end_pkl(final_dict, bigram_uniqueness_strength_pkl_filepath, start)


if __name__ == '__main__':
    main()