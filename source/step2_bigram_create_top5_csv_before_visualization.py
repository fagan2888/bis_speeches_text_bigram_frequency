from config import parameters
from utils import *
import os

# User configuration
max_display_count = 5

output_base_dir = parameters.output_base_dir
bigram_uniqueness_strength_dict_filepath = 'bigram-uniqueness-strength_20191101-02-55-52.pkl'
bigram_uniqueness_strength_dict = load_pkl(os.path.join(output_base_dir, bigram_uniqueness_strength_dict_filepath))

top5_csv_filepath = os.path.join(output_base_dir, get_str_concat('top5-frequency-uniqueness-strength',
                                                                 get_now_time_str()) + '.csv')

printing_order_dict = {0: 'frequency', 1: 'uniqueness', 2: 'strength'}


def main():
    f, wr = start_csv(top5_csv_filepath, csv_delimiter=',')
    wr.writerow(['top5_criteria', 'period_category', 'period', 'bigram', 'frequency', 'uniqueness', 'strength'])
    for printing_order in printing_order_dict.keys():
        for _period_category, _period_dict in bigram_uniqueness_strength_dict.items():
            for _period in sorted(_period_dict.keys()):
                count = 0
                for _bigram, (_freq, _uniqueness, _strength) in sorted(_period_dict[_period].items(),
                                                                       key=lambda t: t[-1][printing_order],
                                                                       reverse=True):  # CONTROL PRINTING ORDER
                    count += 1
                    wr.writerow([printing_order_dict[printing_order], _period_category, _period, _bigram, _freq, _uniqueness, _strength])
                    if count == max_display_count:
                        break

    end_csv(f, top5_csv_filepath)


if __name__ == '__main__':
    main()
