from collections import namedtuple
from pprint import pprint

Event = namedtuple('Event', ['step', 'portfolio', 'action', 'reward'])


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
