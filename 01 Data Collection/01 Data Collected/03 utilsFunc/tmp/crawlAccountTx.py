
import csv
import requests
import sys
import json
from statistics import mean 
from tqdm import tqdm

from threading import Thread

from utilsFunc import make_json_from_csv, parseJson




# Global variables
API_URL = "https://api.etherscan.io/api"
API_KEY = "UF99YFKUDW52Y6TU9I9C2JIBIXP5P6BN6F"

All_data = {}


def main():
    
    args = sys.argv[1:]
    
    num_threads = 4
    
    addr_file_path = args[0]
    
    if (len(args) >= 2):
        num_threads = int(args[1])
    
    # make_json_from_csv(addr_file_path, "addr-list.json")
    
    
    # print("parsing...")
    
    addr_arr, _ = parseJson(addr_file_path)
    
    # # DEBUG
    # print(f'length {len(addr_arr)}')
    # print(f'\naddr[0]:\n {addr_arr[0]}')
    # print(f'\naddr[0]: {addr_arr[0]["address"]} {addr_arr[0]["balance"]}')  
    
    # # Test HTTP request > 10,000
    # print(f'\nTesting addr[0]: "{addr_arr[454]["address"]}"...')
    # getAddrTxData(addr_arr[454]["address"])
    
    # split works in [n] threads and do HTTP request
    # 1) split addr list into [n] equal parts
    # TODO: func for mutiple threads
    threads = []
    
    
    NumOfAddr = len(addr_arr)
    print("Total no. of Address:", NumOfAddr)
    print("No. Of Threads:", num_threads)
    
    # let NumOfAddr = 10 first
    # TODO # *** DEBUG ***
    NumOfAddr = 100
    # ****
    
    prevIdx = 0
    
    # Split the address array into n parts
    for i in range(1, num_threads + 1):
        lenOfEachArr = NumOfAddr // num_threads # e.g., 10 // 4 = 2
        
        i_th_addr_arr = []
        endIdx = 0
        
        if (i != num_threads):
            endIdx = lenOfEachArr * i
        else:
            endIdx = NumOfAddr # ** last one
        
        i_th_addr_arr = addr_arr[prevIdx: endIdx] # e.g., 1st: [0: 0 + 2 * 1] --> 0, 1
                                                  #       2nd: [2: 2 + 2]     --> 2, 3
                                                  #       3rd: [4: 4 + 2]     --> 4, 5
                                                  #       4th: [6: -1]        --> 6, 7, 8, 9  as it is the last thread
                                                  
        prevIdx = endIdx
        
        print(f'{i}th array: {[addr["index"] for addr in i_th_addr_arr]}')
        
        t = Thread(target=testThreads, args=(i_th_addr_arr,))
        
        t.start()
        print(f'Thread {i} started...')
        threads.append(t)
    
    
    # Wait all threads to finish
    for t in threads:
        t.join()
    
    
    print("All threads have finished...")
    
    # sort the data by index
    # All_data_sorted = dict(sorted(All_data.items()))
    
    index_li = list(map(int, list(All_data.keys())))
    
    # print(f"Results:\n {index_li}")
    print(f"length: {len(All_data)}")
    
    
    minIdx = min(index_li)
    MaxIdx = max(index_li)
    print("min idx:", minIdx)
    print("Max idx:", MaxIdx)
    
    
    # write the things to file
    with open(f'test_{minIdx}_{MaxIdx}.json', 'w+') as f:
        f.write(json.dumps(All_data, sort_keys = True, indent = 4))
    
    
    
    
    
    
    return



def getAddrTxData(addr: str):
    # defining a params dict for the parameters to be sent to the API
    PARAMS = {
                'module': 'account',
                'action': 'txlist',
                'address': addr,
                'startblock': 0,
                'endblock': 99999999,
                'page': 1,
                'sort': 'desc',
                'apikey': API_KEY,
              }
    
    # sending get request and saving the response as response object

    r = requests.get(url = API_URL, params = PARAMS)
    
    data = r.json()
    
    return data["result"]
    
    # extracting data in json format
    
    # print("\ntest [0]:",data["result"][0])
    # print("\nlen:", len(data["result"]))

    # # extract average, min, Max gas price and gas fee
    # gas_price_li =[int(g['gasPrice']) for g in data["result"]]
    # print("\ngas price[0]:", gas_price_li[0])
    # print("\n\n** Average Gas Price = %.2f"%(mean(gas_price_li)))
    # print("\n** Max Gas Price = %.2f"%(max(gas_price_li)))
    # print("\n** min Gas Price = %.2f\n"%(min(gas_price_li)))

    # gas_fee_li =[int(g['gasPrice']) * float(g['gasUsed']) for g in data["result"]]
    # print("\n** Average Gas Fee = %.2f"%(mean(gas_fee_li)))
    # print("\n** Max Gas Fee = %.2f"%(max(gas_fee_li)))
    # print("\n** min Gas Fee = %.2f\n\n"%(min(gas_fee_li)))



def testThreads(addr_li: []):
    # print(f'array: {[addr["index"] for addr in addr_li]}')
    
    addrs = [(addr["address"], addr["index"]) for addr in addr_li]
    
    for addr in tqdm(addrs):
        acc_addr = addr[0]
        acc_idx = int(addr[1])
        
        d = getAddrTxData(addr=acc_addr)
        # print(f"Tx: <{acc_addr}>:", len(d))

        All_data[acc_idx] = d


if __name__ == "__main__":

    if (len(sys.argv) < 2):
        print("Usage: python3 crawlAccountTx.py <file_path> <number_of_thread(s) (default: 3)>")
    else:
        main()