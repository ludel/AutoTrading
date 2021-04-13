class Portfolio:

    def __init__(self, initial_account):
        self.initial_account = initial_account
        self.liquidity = initial_account
        self.action_owned = 0
        self.current_price = None

    @property
    def net_worth(self):
        return self.liquidity + (self.action_owned * self.current_price)

    def as_data(self):
        return {'liquidity': self.liquidity, 'action_owned': self.action_owned, 'current_price': self.current_price,
                'net_worth': self.net_worth}
