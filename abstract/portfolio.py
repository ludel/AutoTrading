from dataclasses import dataclass, field


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


@dataclass
class Action:
    current_price: float = 0.00
    owned: int = 0


class ActionPortfolio(dict):

    def __init__(self, codes):
        super().__init__()
        for code in codes:
            self[code] = Action()

    def set_current_price(self, code, price):
        self[code].current_price = price

    def set_owned(self, code, quantity):
        self[code].owned = quantity

    def get_action_price(self, code):
        return self[code].current_price

    def get_action_owned(self, code):
        return self[code].owned


@dataclass
class ImprovedPortfolio:
    initial_account: float
    liquidity: float = field(init=False)
    action_portfolio: ActionPortfolio = field(init=False)

    def __post_init__(self):
        self.liquidity = self.initial_account

    def init_action_portfolio(self, codes_companies):
        self.action_portfolio = ActionPortfolio(codes_companies)

    def buy(self, code, quantity: int):
        price_amount = self.action_portfolio.get_action_price(code) * quantity
        assert price_amount <= self.liquidity

        self.liquidity -= price_amount
        self.action_portfolio.set_owned(code, quantity)

    def sell(self, code, quantity: int):
        assert self.action_portfolio.get_action_owned(code) >= quantity
        price_amount = self.action_portfolio.get_action_price(code) * quantity
        self.liquidity += price_amount
        self.action_portfolio.set_owned(code, quantity)
