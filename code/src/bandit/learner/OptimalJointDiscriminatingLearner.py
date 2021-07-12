from src.bandit.banditEnvironments.JointBanditEnvironment import JointBanditEnvironment


class OptimalJointDiscriminatingLearner:
    def __init__(self, env: JointBanditEnvironment, context_structure):
        # Base info
        self.env = env
        self.n_arms_price = self.env.n_arms_price
        self.n_arms_bid = self.env.n_arms_bid
        self.current_round = 0
        self.security = 0.2
        self.pulled_arms = []

        # Context
        self.context_structure = context_structure
        combs = self.env.get_features_combinations()

        # Arms data per context
        self.future_visits = {c: [[[] for i in range(self.n_arms_bid)]
                                  for j in range(self.n_arms_price)] for c in combs}
        self.purchases = {c: [[[] for i in range(self.n_arms_bid)]
                              for j in range(self.n_arms_price)] for c in combs}
        self.new_clicks = {c: [[[] for i in range(self.n_arms_bid)]
                               for j in range(self.n_arms_price)] for c in combs}
        self.tot_cost_per_click_per_bid = {c: [0 for i in range(self.n_arms_bid)] for c in combs}
        self.tot_auctions_per_bid = {c: [[0] for i in range(self.n_arms_bid)] for c in combs}

        # Recap data
        self.strategies = []

    def get_strategies(self):
        return self.strategies

    def learn(self, n_rounds: int):
        self.round_robin()

        while self.current_round < n_rounds:
            self.learn_one_round()

    def learn_one_round(self):
        strategy_price, strategy_bid = self.choose_next_strategy()
        self.pull_from_env(strategy_price, strategy_bid)

    def pull_from_env(self, strategy_price, strategy_bid):
        # Actual pull
        auctions, new_clicks, purchases, tot_cost_per_clicks, \
            (past_strategies, visits) = self.env.pull_arm_discriminating(strategy_price, strategy_bid)

        # Current round data update
        for comb in self.env.get_features_combinations():
            self.tot_auctions_per_bid[comb][strategy_bid[comb]] += auctions[comb]

            self.new_clicks[comb][strategy_price[comb]][strategy_bid[comb]].append(new_clicks[comb])
            self.purchases[comb][strategy_price[comb]][strategy_bid[comb]].append(purchases[comb])
            self.tot_cost_per_click_per_bid[comb][strategy_bid[comb]] += tot_cost_per_clicks[comb]

        # Past round data update
        if past_strategies[0]:
            price_strat, bid_strat = past_strategies

            for comb in self.env.get_features_combinations():
                old_arm_p = price_strat[comb]
                old_arm_b = bid_strat[comb]
                self.future_visits[comb][old_arm_p][old_arm_b].append(visits[comb])

        # Update context data
        for context in self.context_structure:
            context.merge_all_data(self.future_visits, self.purchases, self.new_clicks, self.tot_cost_per_click_per_bid,
                                   self.tot_auctions_per_bid)

        # Update history
        self.current_round += 1
        self.strategies.append((strategy_price, strategy_bid))

    def choose_next_strategy(self):
        strategy_price = {}
        strategy_bid = {}

        for context in self.context_structure:
            arm_p, arm_b = context.choose_next_arm(self.security, self.current_round)

            for comb in context.features:
                strategy_price[comb] = arm_p
                strategy_bid[comb] = arm_b

        return strategy_price, strategy_bid

    def round_robin(self):
        combs = self.env.get_features_combinations()
        finished = False

        while not finished:
            arm_price = self.current_round % self.n_arms_price
            strategy_price = {comb: arm_price for comb in combs}
            arm_bid = self.current_round % self.n_arms_bid
            strategy_bid = {comb: arm_bid for comb in combs}

            self.pull_from_env(strategy_price, strategy_bid)

            # Check end cycle (trust Jacopo)
            all_price_with_future_visits = all(any(x) for x in self.future_visits[(False, False)])
            bid_without_sample = any(all(not self.new_clicks[(False, False)][p][b] for p in range(self.n_arms_price)) for b in range(self.n_arms_bid))
            finished = all_price_with_future_visits and not bid_without_sample
