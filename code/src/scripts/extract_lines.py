


def main(destination_dir):
    extract_lines(in_file='../bandit/learner/OptimalPriceLearner.py',
                  out_file=destination_dir+'/step3_learning_loop.py',
                  start_delimiter='    # start learning loop\n',
                  end_delimiter='    # end learning loop')

    extract_lines(in_file='../bandit/learner/OptimalPriceLearner.py',
                  out_file=destination_dir + '/step3_projected_profits.py',
                  start_delimiter='    # start projected profits\n',
                  end_delimiter='    # end projected profits')

    extract_lines(in_file='../algorithms.py',
                  out_file=destination_dir + '/step3_simple_class_profit.py',
                  start_delimiter='# start simple class profit\n',
                  end_delimiter='# end simple class profit')

    extract_lines(in_file='../bandit/learner/OptimalPriceLearner.py',
                  out_file=destination_dir + '/step3_estimates.py',
                  start_delimiter='    # start compute estimates\n',
                  end_delimiter='    # end compute estimates')

    extract_lines(in_file='../bandit/learner/OptimalPriceLearner.py',
                  out_file=destination_dir + '/step3_conversion_rates.py',
                  start_delimiter='    # start conversion rates\n',
                  end_delimiter='    # end conversion rates')

    extract_lines(in_file='../bandit/learner/ucb/UCBOptimalPriceLearner.py',
                  out_file=destination_dir + '/step3_ucb.py',
                  start_delimiter='    # start conv rate\n',
                  end_delimiter='    # end conv rate')

    extract_lines(in_file='../bandit/learner/ts/TSOptimalPriceLearner.py',
                  out_file=destination_dir + '/step3_ts_cr.py',
                  start_delimiter='    # start conversion rate\n',
                  end_delimiter='    # end conversion rate')

    extract_lines(in_file='../bandit/learner/ts/TSOptimalPriceLearner.py',
                  out_file=destination_dir + '/step3_ts_betas.py',
                  start_delimiter='    # start update betas\n',
                  end_delimiter='    # end update betas')

    extract_lines(in_file='../bandit/learner/OptimalPriceDiscriminatingLearner.py',
                  out_file=destination_dir + '/step4_learning_loop.py',
                  start_delimiter='    # start learning loop\n',
                  end_delimiter='    # end learning loop')

    extract_lines(in_file='../bandit/learner/OptimalPriceDiscriminatingLearner.py',
                  out_file=destination_dir + '/step4_choose_next_strategy.py',
                  start_delimiter='    # start choose next strategy\n',
                  end_delimiter='    # end choose next strategy')

    extract_lines(in_file='../bandit/learner/OptimalPriceDiscriminatingLearner.py',
                  out_file=destination_dir + '/step4_choose_next_strategy_explorative.py',
                  start_delimiter='    # start choose next explorative\n',
                  end_delimiter='    # end choose next explorative')

    extract_lines(in_file='../bandit/learner/OptimalPriceDiscriminatingLearner.py',
                  out_file=destination_dir + '/step4_update_contexts.py',
                  start_delimiter='    # start update context\n',
                  end_delimiter='    # end update context')

    extract_lines(in_file='../bandit/learner/OptimalPriceDiscriminatingLearner.py',
                  out_file=destination_dir + '/step4_convenient_split.py',
                  start_delimiter='    # start convenient splits\n',
                  end_delimiter='    # end convenient splits')

    extract_lines(in_file='../bandit/learner/OptimalPriceDiscriminatingLearner.py',
                  out_file=destination_dir + '/step4_possible_split.py',
                  start_delimiter='    # start possible splits\n',
                  end_delimiter='    # end possible splits')

    extract_lines(in_file='../bandit/context/Context.py',
                  out_file=destination_dir + '/step4_context_next_arm.py',
                  start_delimiter='    # start context next arm\n',
                  end_delimiter='    # end context next arm')

    extract_lines(in_file='../bandit/learner/ucb/UCBOptimalPriceDiscriminatingLearner.py',
                  out_file=destination_dir + '/step4_ucb_disc.py',
                  start_delimiter='# start ucbdisc\n',
                  end_delimiter='# end ucbdisc')

    extract_lines(in_file='../bandit/context/UCBContext.py',
                  out_file=destination_dir + '/step4_ucbcontext_projection.py',
                  start_delimiter='    # start projection\n',
                  end_delimiter='    # end projection')

    extract_lines(in_file='../bandit/learner/ts/TSOptimalPriceDiscriminatingLearner.py',
                  out_file=destination_dir + '/step4_ts_disc.py',
                  start_delimiter='# start ts disc\n',
                  end_delimiter='# end ts disc')

    extract_lines(in_file='../bandit/context/TSContext.py',
                  out_file=destination_dir + '/step4_tscontext_projection.py',
                  start_delimiter='    # start projection\n',
                  end_delimiter='    # end projection')


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
