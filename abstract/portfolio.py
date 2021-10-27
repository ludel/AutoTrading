from typing import Callable, Optional

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
    code: str
    price: float = 0.0
    owned: int = 0

    def get_total_value(self):
        return self.price * self.owned

    def __repr__(self):
        return f"Action({self.code}, price={self.price}, owned={self.owned}, value={self.get_total_value()})"


class ActionPortfolio(dict):

    def __init__(self, codes):
        super().__init__()
        for code in codes:
            self[code] = Action(code)

    def set_current_price(self, code, price):
        self[code].price = price

    def set_owned(self, code, quantity):
        self[code].owned = quantity

    def get_action_price(self, code):
        return self[code].price

    def get_action_owned(self, code):
        return self[code].owned

    def get_net_worth(self):
        return sum(action.get_total_value() for action in self.values())

    def export_prices(self):
        return {v.code: v.price for v in self.values()}

    def summary(self):
        for key, value in self.items():
            print(key, value, sep=' => ')


@dataclass
class ImprovedPortfolio:
    initial_account: float
    liquidity: float = field(init=False)
    action_portfolio: ActionPortfolio = field(init=False)
    sell_callback: Optional[Callable] = None
    buy_callback: Optional[Callable] = None

    def __post_init__(self):
        self.liquidity = self.initial_account

    def init_action_portfolio(self, codes_companies: str):
        self.action_portfolio = ActionPortfolio(codes_companies)

    @property
    def net_worth(self):
        return self.liquidity + self.action_portfolio.get_net_worth()

    def buy(self, code, quantity: int):
        price_amount = self.action_portfolio.get_action_price(code) * quantity
        assert price_amount <= self.liquidity

        self.liquidity -= price_amount
        self.action_portfolio.set_owned(code, quantity)

        if self.buy_callback:
            self.buy_callback(code, quantity, self)

    def sell(self, code):
        current_position = self.action_portfolio.get_action_owned(code)
        price_amount = self.action_portfolio.get_action_price(code) * current_position
        self.liquidity += price_amount
        self.action_portfolio.set_owned(code, 0)

        if self.sell_callback:
            self.sell_callback(code, current_position, self)

    def set_action_price(self, code, price: float):
        self.action_portfolio.set_current_price(code, price)

    def get_action_price(self, code):
        return self.action_portfolio.get_action_price(code)

    def get_position(self, code):
        return self.action_portfolio[code]

    def summary(self):
        print('Initial {} - Liquidity {} - Net Worth {}'.format(self.initial_account, self.liquidity, self.net_worth))
