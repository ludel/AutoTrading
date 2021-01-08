import csv
import getpass
from time import sleep

import degiroapi
from degiroapi.product import Product

password = getpass.getpass()
degiro = degiroapi.DeGiro()
degiro.login("ludel33", password)

products_all_cac = [Product(p) for p in degiro.get_stock_list(5, 886)]
size_list = len(products_all_cac)
fieldnames = ['name', 'isin', 'symbol', 'date', 'price']

with open('../data/stocks_cac_40.csv', 'w') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=',')
    writer.writeheader()

    for index, product in enumerate(products_all_cac):
        prices = degiro.real_time_price(product.id, degiroapi.Interval.Type.Max)[1]
        for day_data in prices['data']:
            writer.writerow({
                'name': product.name,
                'isin': product.isin,
                'symbol': product.symbol,
                'date': day_data[0],
                'price': day_data[1],
            })

        print('{}/{}'.format(index, size_list))
        sleep(2)
