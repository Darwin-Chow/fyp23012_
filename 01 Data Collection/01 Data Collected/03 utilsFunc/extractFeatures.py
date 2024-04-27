
import json
import csv
from statistics import mean 

# def convertJsonToCsv(jsonData):
#     with open('address_summary_complete.csv', 'w+') as csvF:
#         csv_writer = csv.writer(csvF)
#         count = 0
#         for data in jsonData:
#             if count == 0:
#                 header = data.keys()
#                 csv_writer.writerow(header)
#                 count += 1
#             csv_writer.writerow(data.values())
    

addr_tx_list = []
with open('response.json', 'r') as jsonfile:
    x = jsonfile.read()
    # print(x)
    addr_tx_list = json.loads(x)

# print(addr_sum_list)
print("\ntest [0]:",addr_tx_list["result"][0])
print("\nlen:", len(addr_tx_list["result"]))

# extract average, min, Max gas price and gas fee
gas_price_li =[int(g['gasPrice']) for g in addr_tx_list["result"]]
print("\ngas price[0]:", gas_price_li[0])
print("\n\n** Average Gas Price = %.2f"%(mean(gas_price_li)))
print("\n** Max Gas Price = %.2f"%(max(gas_price_li)))
print("\n** min Gas Price = %.2f\n"%(min(gas_price_li)))

gas_fee_li =[int(g['gasPrice']) * float(g['gasUsed']) for g in addr_tx_list["result"]]
print("\n** Average Gas Fee = %.2f"%(mean(gas_fee_li)))
print("\n** Max Gas Fee = %.2f"%(max(gas_fee_li)))
print("\n** min Gas Fee = %.2f"%(min(gas_fee_li)))


