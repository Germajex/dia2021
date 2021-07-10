


def main(destination_dir):
    extract_lines(in_file='../bandit/OptimalPriceLearner.py',
                  out_file=destination_dir+'/step3_learning_loop.py',
                  start_delimiter='    # start learning loop\n',
                  end_delimiter='    # end learning loop')

    extract_lines(in_file='../bandit/OptimalPriceLearner.py',
                  out_file=destination_dir + '/step3_projected_profits.py',
                  start_delimiter='    # start projected profits\n',
                  end_delimiter='    # end projected profits')

    extract_lines(in_file='../algorithms.py',
                  out_file=destination_dir + '/step3_simple_class_profit.py',
                  start_delimiter='# start simple class profit\n',
                  end_delimiter='# end simple class profit')

    extract_lines(in_file='../bandit/OptimalPriceLearner.py',
                  out_file=destination_dir + '/step3_estimates.py',
                  start_delimiter='    # start compute estimates\n',
                  end_delimiter='    # end compute estimates')

    extract_lines(in_file='../bandit/OptimalPriceLearner.py',
                  out_file=destination_dir + '/step3_conversion_rates.py',
                  start_delimiter='    # start conversion rates\n',
                  end_delimiter='    # end conversion rates')

    extract_lines(in_file='../bandit/UCBOptimalPriceLearner.py',
                  out_file=destination_dir + '/step3_ucb.py',
                  start_delimiter='    # start conv rate\n',
                  end_delimiter='    # end conv rate')

    extract_lines(in_file='../bandit/TSOptimalPriceLearner.py',
                  out_file=destination_dir + '/step3_ts_cr.py',
                  start_delimiter='    # start conversion rate\n',
                  end_delimiter='    # end conversion rate')

    extract_lines(in_file='../bandit/TSOptimalPriceLearner.py',
                  out_file=destination_dir + '/step3_ts_betas.py',
                  start_delimiter='    # start update betas\n',
                  end_delimiter='    # end update betas')


def extract_lines(in_file, out_file, start_delimiter, end_delimiter):
    with open(in_file, 'r', encoding='utf8') as inf:
        lines = ''.join(inf.readlines())
        start = lines.index(start_delimiter)
        end = lines.index(end_delimiter, start)
        lines = lines[start+len(start_delimiter):end]
        with open(out_file, 'w', encoding='utf8') as of:
            of.write(lines.rstrip())


if __name__ == "__main__":
    main(destination_dir='../../../report/code')
