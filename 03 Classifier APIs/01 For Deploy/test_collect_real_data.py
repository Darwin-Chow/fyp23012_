
# HTTP request & file
import requests
# import sys
import json

# extract Features
import numpy as np
from statistics import mean 
from statistics import stdev

# crawl Etherscan using ChromeDriver
# seems not work --> need load chromeDriver frmo database

# Other Function & Variables
from test_predict import classifyAccount
from utilsTools import ETH_accounts, Etherscan_API, Oklink_API, ExplorerAPI


def classify(acc_addr):
    
    # print(f"\n--- Testing {acc_type.name} account ---")
    print(f"\taddress: {acc_addr}\n")
    
    # Step 1: get Acc Summary from Oklink API
    acc_summary = retrieveAccSummary(acc_addr)
    
    # DEBUG pt01 (DONE)
    
    # Step 2a: get normal transaction list
    norm_tx_li = retrieveAccTx(acc_addr, isNormal=True)
    
    
    # Step 2b: get internal transaction list (if presented)
    internal_tx_li = retrieveAccTx(acc_addr, isNormal=False)
    
    
    if (len(norm_tx_li) == 0 and len(internal_tx_li) == 0):
        print("\t-- Zero Transaction Account --\n")
        print("\t-- Exiting... --\n")
        acc_summary['isZeroTx'] = 1
        
        return (acc_summary, "NaN")
    
    
    # TODO
    # Step 3: get Highest, lowest Balance & date (using webdriver) --> Etherscan.io
    extra_feature = {}
    extra_feature = retrieveAccExtraData(acc_addr)  

    
    
    # Step 4: Extract & Combine ALL Features
    acc_all_features = combineAndExtract(acc_summary, norm_tx_li, internal_tx_li, extra_feature)
    
    print(f"\n -- ALL Features (len: {len(acc_all_features)}): --\n", json.dumps(acc_all_features, sort_keys = False, indent = 4), "\n")
    
    
    addBackTimeDiff_features(acc_all_features)
    
    # Step 5: Save the data to firebase & pack the data for input for Classifier
    # TODO: next 
    prob_percent = classifyAccount(acc_all_features)[0]
    
    print("\n---- Summary ----\n")
    print(f"\tInput Address: {acc_addr}\n")
    # print(f"\tType of Account: '{acc_type.name}'\n")
    print(f"\tProbability as Illicit: {prob_percent}%\n")
    print("\n---- ------- ----\n")
    
    
    
    
    return (acc_all_features, prob_percent)



def main():
    
    # dummary account address for testing
    acc_type = ETH_accounts.NORMAL2
    
    # acc_type_name = acc_type.name
    acc_addr = acc_type.value
    
    print(f"\n--- Testing {acc_type.name} account ---")
    print(f"\taddress: {acc_addr}\n")
    
    # Step 1: get Acc Summary from Oklink API
    acc_summary = retrieveAccSummary(acc_addr)
    
    # DEBUG pt01 (DONE)
    
    # Step 2a: get normal transaction list
    norm_tx_li = retrieveAccTx(acc_addr, isNormal=True)
    
    
    # Step 2b: get internal transaction list (if presented)
    internal_tx_li = retrieveAccTx(acc_addr, isNormal=False)
    
    
    if (len(norm_tx_li) == 0 and len(internal_tx_li) == 0):
        print("\t-- Zero Transaction Account --\n")
        print("\t-- Exiting... --\n")
        return
    
    # DEBUG pt02 (DONE)
    # print("--- Normal Transaction ---\n")
    # if (len(norm_tx_li) > 0):
    #     print(f"\t# {len(norm_tx_li)} Transaction #\n\t", norm_tx_li[0], "\n")
    # else:
    #     print("\t0 normal transaction", "\n")
        
    
    # print("--- Internal Transaction ---\n")
    # if  (len(internal_tx_li) > 0):
    #     print(f"\t# {len(internal_tx_li)} Transaction #\n\t", internal_tx_li[0], "\n")
    # else:
    #     print("\t0 internal transaction", "\n")
    
    # return
    
    # TODO
    # Step 3: get Highest, lowest Balance & date (using webdriver) --> Etherscan.io
    extra_feature = {}
    extra_feature = retrieveAccExtraData(acc_addr)  
    
    # DEBUG pt03
    # return  
    
    
    
    # Step 4: Extract & Combine ALL Features
    acc_all_features = combineAndExtract(acc_summary, norm_tx_li, internal_tx_li, extra_feature)
    
    print(f"\n -- ALL Features (len: {len(acc_all_features)}): --\n", json.dumps(acc_all_features, sort_keys = False, indent = 4), "\n")
    
    
    addBackTimeDiff_features(acc_all_features)
    
    # Step 5: Save the data to firebase & pack the data for input for Classifier
    # TODO: next 
    prob_percent = classifyAccount(acc_all_features)[0]
    
    print("\n---- Summary ----\n")
    print(f"\tInput Address: {acc_addr}\n")
    print(f"\tType of Account: '{acc_type.name}'\n")
    print(f"\tProbability as Illicit: {prob_percent}%\n")
    print("\n---- ------- ----\n")
    
    
    return



def retrieveAccSummary(acc_addr: str) -> {}:
    
    # print INFO
    print("\n### Collecting Address Summary... ###\n")
    
    
    API_URL = Oklink_API.URL
    API_keys = Oklink_API.key_li
    
    # set headers
    HEADERS = {
        'Content-Type': 'application/json',
        'Ok-Access-Key': API_keys[0],
    }
    
        
    # defining a params dict for the parameters to be sent to the API
    PARAMS = {
        'chainShortName': 'eth',
        'address': acc_addr,
    }
    
    success = False
    attemptCnt = 0

    while (not success):
        # sending get request and saving the response as response object
        
        try:
            r = requests.get(url = API_URL, params = PARAMS, headers=HEADERS)
                    
            data = r.json()
            
            # --- DEBUG ---
            # print("test code:", data['code'][0])
            # print("test data:", data['data'][0])
            # print("test:\n", data)
            # --- DEBUG ---
            
            
            # 1st Status: Success
            if (data['code'][0] == '0' and len(data['data']) > 0):
                addrSummaryData = data['data'][0]
                
                success = True
                
                print("\t-- Success! --\n\t", addrSummaryData, "\n")
                
                return addrSummaryData
                
            
            # 2nd Status: No records found
            elif (data['code'][0] == '0' and len(data['data']) == 0):
                print("-- Success (no Tx)! --\n", addrSummaryData, "\n")
                success = True
                return {}
                

            # 3rd Status: Error Occurs
            else:
                print("-- Failure! --\n")
                attemptCnt += 1
                if (attemptCnt >= 5):
                    return {}
                
        except Exception as e:
            print("Error:", e)
            attemptCnt += 1
            if (attemptCnt >= 5):
                return {}
    
    
    return {}



def retrieveAccTx(acc_addr: str, isNormal: bool) -> [{}]:
    
    API_URL = Etherscan_API.URL
    API_key_list = Etherscan_API.key_li
    
    action = ''
    
    if (isNormal):
        # TODO : do normal tx stuff
        action = 'txlist'
        # print INFO
        print("\n### Collecting Normal Transaction Summary... ###\n")
        
    else:
        # TODO: do internal tx stuff
        action = 'txlistinternal'
        
        # print INFO
        print("\n### Collecting Internal Transaction Summary... ###\n")

    
    data = -1
    k = 0

    while (data == -1 or data["result"] == "Max rate limit reached"):
        CUR_API_KEY = API_key_list[k]

        # defining a params dict for the parameters to be sent to the API
        PARAMS = {
                    'module': 'account',
                    'action': action,
                    'address': acc_addr,
                    'startblock': 0,
                    'endblock': 99999999,
                    'page': 1,
                    'sort': 'desc',
                    'apikey': CUR_API_KEY,
        }
            
        # sending get request and saving the response as response object

        r = requests.get(url = API_URL, params = PARAMS)
            
        data = r.json()

        k += 1

        if (k >= 2):
            k = 0

        
        
    
    return data["result"]
        
    
    
    return 


def retrieveAccExtraData(acc_addr: str) -> []:
    
    extra_features = {
        'highestBalance': "NaN",
        'lowestBalance': "NaN",
        'time_diff_between_max_balance_and_first_tx': "NaN",
        'time_diff_between_max_balance_and_last_tx': "NaN",
        'time_diff_between_min_balance_and_first_tx': "NaN",
        "time_diff_between_min_balance_and_last_tx": "NaN"
    }
    
    return extra_features


#  TODO: 
# 1) extract features from transaction list
# 2) combine all features as dict
def combineAndExtract(acc_summary, norm_tx_li, internal_tx_li, extra_feature):
    
    All_features = {}
    addr_this = acc_summary['address']
    addr_all_tx_cnt = int(acc_summary['transactionCount'])
    
    norm_and_internal_tx_li = [norm_tx_li, internal_tx_li]
    
    
    # extract TX features
    
    acc_norm_features = {}
    acc_internal_features = {}
    
    for k in range(2):
        
        # when k = 0 --> normal; k = 1 --> internal tx
        tx_li = norm_and_internal_tx_li[k]
        
        if (k == 0):
            print("\n### Extracting Normal Transaction Features... ###\n")
        else:
            print("\n### Extracting Internal Transaction Features... ###\n")

        
        # 1a) feature extraction in normal TXs
        addr_tx_send_di = {}
        addr_tx_receive_di = {}
        tx_recv_time_li = []
        tx_send_time_li = []
        tx_time_li = []
        tx_recv_val_li = []
        tx_send_val_li = []
        
        # features to be extracted
        Tx_count = 0
        out_tx = 0
        out_tx_percent = 0
        in_tx = 0
        in_tx_percent = 0

        # 
        uniq_receive_address_num = 0
        uniq_send_address_num = 0
        
        avg_time_between_send_tx = 0
        avg_time_between_recv_tx = 0
        avg_time_between_tx = 0

        # 
        zero_val_tx_cnt = 0
        zero_val_send_tx_cnt = 0
        zero_val_recv_tx_cnt = 0
        


        # get number of normal TX
        Tx_count = len(tx_li)
        
        print(f"\n\t-- Transaction Count : {Tx_count} --\n")

        
        # print(f"\n****  DEBUG [{addr_idx}]: <{addr_this}> ****")
        
        # get number of normal tx send from 'addr'
        # get number of normal tx receive from 'addr'
        for tx in tx_li:
            
            
            # OUT (send)
            if (tx['from'] == addr_this):
                out_tx += 1
                
                # get min send amount
                addr_tx_send_di.setdefault(tx['to'], 0)
                addr_tx_send_di[tx['to']] += 1
                
                if (tx['isError'] == "1"):
                    continue

                # get maximum/ min time diff between send & receive
                tx_send_time_li.append(int(tx['timeStamp']) / 1000)
                tx_time_li.append(int(tx['timeStamp']) / 1000)
                
                # get max send
                tx_send_val_li.append(float(tx['value']))
                
                # get zero value tx
                if (int(tx['value']) == 0):
                    zero_val_send_tx_cnt += 1
                    zero_val_tx_cnt += 1
                    
            
            # IN (receive)
            elif (tx['to'] == addr_this):
                in_tx += 1
                
                # No Transaction fee for receiving (IN)
                addr_tx_receive_di.setdefault(tx['from'], 0)
                addr_tx_receive_di[tx['from']] += 1
                
                # : get time diff
                tx_recv_time_li.append(int(tx['timeStamp']) / 1000 )
                tx_time_li.append(int(tx['timeStamp']) / 1000)
                
                #  get max recv
                tx_recv_val_li.append(float(tx['value']))
                
                # get zero value tx
                if (int(tx['value']) == 0):
                    zero_val_recv_tx_cnt += 1
                    zero_val_tx_cnt += 1

        # TODO: convert historical balacen to Ether (val / (10^18))
        

        # calculate uniq address send and receive
        for v in addr_tx_receive_di.values():
            if (v == 1):
                uniq_receive_address_num += 1
        for v in addr_tx_send_di.values():
            if (v == 1):
                uniq_send_address_num += 1

        
        # get max, min, avg, std of send or recv values
        
        if (len(tx_send_val_li) == 0):
            max_val_send = min_val_send = mean_val_send = "NaN"
        else:
            max_val_send = max(tx_send_val_li) / (10 ** 18)
            min_val_send = min(tx_send_val_li) / (10 ** 18)
            mean_val_send = mean(tx_send_val_li) / (10 ** 18)
        
        if (len(tx_send_val_li) > 1):
            stdev_val_send = stdev(tx_send_val_li) / (10 ** 18)
        else:
            stdev_val_send = 0
        
        
        # receive
        if (len(tx_recv_val_li) == 0):
            max_val_recv = min_val_recv = mean_val_recv = "NaN"
        else:
            max_val_recv = max(tx_recv_val_li) / (10 ** 18)
            min_val_recv = min(tx_recv_val_li) / (10 ** 18)
            mean_val_recv = mean(tx_recv_val_li) / (10 ** 18)
        
        if (len(tx_recv_val_li) > 1):
            stdev_val_recv = stdev(tx_recv_val_li) / (10 ** 18)
        else:
            stdev_val_recv = 0
        
        
        
        # calculate time differences
        tx_time_li.reverse()
        a = np.array(tx_time_li)
        diff_li = np.diff(a)
        avg_time_between_tx = mean(diff_li) if (len(diff_li) > 1) else "NaN"
        
        tx_send_time_li.reverse()
        a = np.array(tx_send_time_li)
        diff_li = np.diff(a)
        avg_time_between_send_tx = mean(diff_li) if (len(diff_li) > 1) else "NaN"
        
        tx_recv_time_li.reverse()
        a = np.array(tx_recv_time_li)
        diff_li = np.diff(a)
        avg_time_between_recv_tx = mean(diff_li) if (len(diff_li) > 1) else "NaN"
        


        
        # 
        
        if (Tx_count == 0):
            out_tx_percent = in_tx_percent = 0
            
        else:
            out_tx_percent = out_tx / (float(Tx_count))
            in_tx_percent = in_tx / (float (Tx_count) )

        
        # Specific for Normal
        if (k == 0):
            
            # get gas price (in gwei), fee in AVG, min, MAX
            # extract average, min, Max gas price and gas fee (in ETH)
            gas_price_li =[int(g['gasPrice']) / (10 ** 9) for g in tx_li]
            tx_fee_li =[(int(g['gasPrice']) * float(g['gasUsed'])) / (10 ** 18) for g in tx_li]
            
        
            
            # gas price
            if (len(gas_price_li) == 0):
                mean_gas_price = max_gas_price = min_gas_price = stdev_gas_price = "NaN"
                mean_transaction_fee = max_transaction_fee = min_transaction_fee = stdev_transaction_price = "NaN"

            else:
                mean_gas_price = mean(gas_price_li) if len(gas_price_li) > 1 else gas_price_li[0]
                max_gas_price = max(gas_price_li)
                min_gas_price = min(gas_price_li)
                stdev_gas_price = stdev(gas_price_li) if len(gas_price_li) > 1 else "NaN"
            
                # gas fee
                mean_transaction_fee = mean(tx_fee_li) if len(tx_fee_li) > 1 else tx_fee_li[0]
                max_transaction_fee = max(tx_fee_li)
                min_transaction_fee = min(tx_fee_li)
                stdev_transaction_price = stdev(tx_fee_li) if len(tx_fee_li) > 1 else "NaN"
            
            
            
            acc_norm_features = {
            
                # Basic Attributes
                "address": addr_this,
                
                # Transaction Count
                # "total_transaction_count": addr_all_tx_cnt,
                "num_of_normal_transaction": Tx_count,
                "out_transaction_percent": out_tx_percent,
                "in_transaction_percent": in_tx_percent,
                
                # Send Value
                "max_val_send": max_val_send,
                "min_val_send": min_val_send,
                "mean_val_send": mean_val_send,
                "stdev_val_send": stdev_val_send,
                
                # Receive Value
                "max_val_recv": max_val_recv,
                "min_val_recv": min_val_recv,
                "mean_val_recv": mean_val_recv,
                "stdev_val_recv": stdev_val_recv,
                
                # Gas Price
                "max_gas_price": max_gas_price,
                "min_gas_price": min_gas_price,
                "mean_gas_price": mean_gas_price,
                "stdev_gas_price": stdev_gas_price,
                
                # Transaction fee
                "mean_transaction_fee": mean_transaction_fee,
                "max_transaction_fee": max_transaction_fee,
                "min_transaction_fee": min_transaction_fee,
                "stdev_transaction_price": stdev_transaction_price,
                
                # Unique address send/ receive
                "uniq_send_address_num": uniq_send_address_num,
                "uniq_receive_address_num": uniq_receive_address_num,
                
                # zero value transaction
                "zero_val_tx_num": zero_val_tx_cnt,
                "zero_val_send_tx_num": zero_val_send_tx_cnt,
                "zero_val_recv_tx_num": zero_val_recv_tx_cnt,
                
                # time between transactions
                "mean_time_between_tx": avg_time_between_tx,
                "mean_time_between_send_tx": avg_time_between_send_tx,
                "mean_time_between_recv_tx": avg_time_between_recv_tx,
            }
        
        else:
            
            gas_li =[int(g['gas']) / (10 ** 9) for g in tx_li]
            max_gas = min_gas = mean_gas = stdev_gas = 0
            
            if (len(gas_li) == 0):
                max_gas = min_gas = mean_gas = stdev_gas = "NaN"
            else:
                # gas price
                max_gas =  max(gas_li)
                min_gas = min(gas_li)
                mean_gas = mean(gas_li) if len(gas_li) > 1 else gas_li[0]
                stdev_gas = stdev(gas_li) if len(gas_li) > 1 else "NaN"
                
            acc_internal_features = {
                    
                # Transaction Count
                "num_of_internal_transaction": Tx_count,
                "internal_out_transaction_percent": out_tx_percent,
                "internal_in_transaction_percent": in_tx_percent,
                
                # Send Value
                "internal_max_val_send": max_val_send,
                "internal_min_val_send": min_val_send,
                "internal_mean_val_send": mean_val_send,
                "internal_stdev_val_send": stdev_val_send,
                
                # Receive Value
                "internal_max_val_recv": max_val_recv,
                "internal_min_val_recv": min_val_recv,
                "internal_mean_val_recv": mean_val_recv,
                "internal_stdev_val_recv": stdev_val_recv,
                
                # Gas Price
                "internal_max_gas": max_gas,
                "internal_min_gas": min_gas,
                "internal_mean_gas": mean_gas,
                "internal_stdev_gas_price": stdev_gas,
                
                
                # Unique address send/ receive
                "internal_uniq_send_address_num": uniq_send_address_num,
                "internal_uniq_receive_address_num": uniq_receive_address_num,
                
                # zero value transaction
                "internal_zero_val_tx_num": zero_val_tx_cnt,
                "internal_zero_val_send_tx_num": zero_val_send_tx_cnt,
                "internal_zero_val_recv_tx_num": zero_val_recv_tx_cnt,
                
                # time between transactions
                "internal_mean_time_between_tx": avg_time_between_tx,
                "internal_mean_time_between_send_tx": avg_time_between_send_tx,
                "internal_mean_time_between_recv_tx": avg_time_between_recv_tx,
                
            }
            
    
    
    print("\n -- Normal Tx Features: --\n", json.dumps(acc_norm_features, sort_keys = False, indent = 4))
    print("\n -- Internal Tx Features: --\n", json.dumps(acc_internal_features, sort_keys = False, indent = 4) )
    
    # new 
    All_features['total_transaction_count'] = acc_norm_features['num_of_normal_transaction'] + acc_internal_features['num_of_internal_transaction']
    
    # old to float
    acc_summary['balance'] = float(acc_summary['balance'])
    acc_summary['transactionCount'] = float(acc_summary['transactionCount'])
    acc_summary['sendAmount'], acc_summary['receiveAmount'] = float(acc_summary['sendAmount']), float(acc_summary['receiveAmount']), 
    acc_summary['tokenAmount'] = float(acc_summary['tokenAmount']) if (acc_summary['tokenAmount'] != '') else 'NaN'
    
    acc_summary['totalTokenValue'] = float(acc_summary['totalTokenValue']) if (acc_summary['totalTokenValue'] != '') else 'NaN'
    acc_summary['firstTransactionTime'], acc_summary['lastTransactionTime'] = float(acc_summary['firstTransactionTime']), float(acc_summary['lastTransactionTime'])
    
    # Combine All features
    updateMutipleDict(All_features, acc_summary, acc_norm_features, acc_internal_features, extra_feature)
    
    
    
    # print(f"\n\t -- ALL Features (len: {len(All_features)}): --\n\t", All_features, "\n")
    return All_features
    


def addBackTimeDiff_features(all_features):
    all_features['time_between_first_and_last_tx'] = (all_features['lastTransactionTime'] - all_features['firstTransactionTime']) / 1000
    
    

def updateMutipleDict(di, *dicts) -> None:
    for od in dicts:
        di.update(od)


if __name__ == "__main__":
    classify("0x0a52ecaa61268c6a5cf9cd6b1378531a4672601b")
    # main()