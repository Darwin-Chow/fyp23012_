
import json
import csv

def convertJsonToCsv(jsonData):
    with open('address_summary_complete.csv', 'w+') as csvF:
        csv_writer = csv.writer(csvF)
        count = 0
        for data in jsonData:
            if count == 0:
                header = data.keys()
                csv_writer.writerow(header)
                count += 1
            csv_writer.writerow(data.values())
    

addr_sum_list = []
with open('input.json', 'r') as jsonfile:
    x = jsonfile.read()
    # print(x)
    addr_sum_list = json.loads(json.loads(x))

# print(addr_sum_list)
print("\ntest [0]:",addr_sum_list[0])
# print("\naddress [0]:", addr_sum_list[0]['address'])
print("\nlen:", len(addr_sum_list))


# convertJsonToCsv(addr_sum_list)

with open('transactionTime.csv', 'w+') as txf:
    
    txf.write('firstTransactionTime,lastTransactionTime')
    txf.write('\n')
    
    for i in range(len(addr_sum_list)):
        firstTx = addr_sum_list[i]['firstTransactionTime']
        lastTx = addr_sum_list[i]['lastTransactionTime']
        
        txf.write(firstTx + ',' + lastTx)
        txf.write('\n')
        
    
    
        



# addr_raw = []
# with open('address_complete.json', 'r') as f:
#     addr_raw = json.load(f)

# print(addr_raw[0]['Address'])
# print("\nlen:", len(addr_raw))

# success = True
# for i in range(0, len(addr_sum_list)):
#     if (addr_sum_list[i]['address'] != addr_raw[i]['Address']):
#         # print("!!! ERROR !!!")
#         success = False

# if (success):
#     print("*** SUCCESS ***")



# with open('output.txt', 'w+') as f:
#     for item in addr_sum_list:
#         f.write(item['address'])
#         f.write("\n")

