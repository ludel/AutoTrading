from collections import namedtuple
from pprint import pprint

Event = namedtuple('Event', ['step', 'portfolio', 'action', 'reward'])
EventImproved = namedtuple('Event', ['day', 'sell_index', 'buy_index', 'portfolio_value', 'portfolio_action', 'reward'])


class History(list):
    def __init__(self):
        super(History, self).__init__()
        self.last_buy_step = None

    def add_event(self, element: Event):
        self.append(element)

    def show_action(self):
        for event in self:
            print(event.action)

    def print_all(self):
        pprint(self)


class HistoryImprove(list):
    def __init__(self):
        super(HistoryImprove, self).__init__()

    def get_all_price_by_code(self, code, initial_amount):
        prices = []
        quantity_action = None
        for event in self:
            for company_code, price in event.portfolio_action.items():
                if company_code == code:
                    if not quantity_action:
                        quantity_action = initial_amount / price
                    prices.append(price * quantity_action)
        return prices

    def add(self, element: EventImproved):
        self.append(element)

    def get_first_prices(self):
        return list(self[0].portfolio_action.values())

    def get_all_code(self):
        return list(self[0].portfolio_action.keys())

    def get_all_rewards(self):
        return [h.reward for h in self]

    def get_portfolio_values(self):
        return [h.portfolio_value for h in self]

    def get_portfolio_action_owned(self, owned_actions):
        sum_price = []
        for event in self:
            sum_price.append(sum(owned_actions * list(event.portfolio_action.values())))

        return sum_price
