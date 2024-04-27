
import json
import csv
from statistics import mean 
from statistics import stdev
import math
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../03 utilsFunc')

from utilsFunc import make_json_from_csv, make_csv_from_JSON_data, parseJson, getAddrList

TEST_FILE = 'normalTx_1-10.json'

REAL_FILE = 'normalTx_1_13941.json'

REAL_FILE_internal = 'internal_tx_1_13941.json'

NAN_VAL = "-1"

All_addr_features_li = []


def main():
    
    
    # 1) get address list to be collected
    addr_list = getAddrList()
    
    print("number of address:", len(addr_list))

    
    # 2) get transaction list of all addresses
    addr_TX_di, addr_len = parseJson(REAL_FILE)
    
    # print(f"[1]:\n{addr_di['1']}")
    
    print("length:", addr_len)
    
    
    # 3) feature extraction in normal TXs
    for addr_summary in tqdm(addr_list):
    
        addr_idx = addr_summary['index']
        addr_this = addr_summary['address']
        addr_cur_balance = float(addr_summary['balance'])
        addr_all_tx_cnt = int(addr_summary['transactionCount'])
        
        # For REF
        addr_balance = float(addr_summary['balance'])
        
        addr_tx_send_di = {}
        addr_tx_receive_di = {}
        tx_recv_time_li = []
        tx_send_time_li = []
        tx_time_li = []
        tx_recv_val_li = []
        tx_send_val_li = []
        
        # features to be extracted
        norm_Tx_count = 0
        out_tx = 0
        out_tx_percent = 0
        in_tx = 0
        in_tx_percent = 0

        
        # TODO
        uniq_receive_address_num = 0
        uniq_send_address_num = 0
        
        avg_time_between_send_tx = 0
        stdev_time_between_send_tx = 0
        
        avg_time_between_recv_tx = 0
        stdev_time_between_recv_tx = 0
        
        avg_time_between_tx = 0
        stdev_time_between_tx = 0
        
        
        # 
        zero_val_tx_cnt = 0
        zero_val_send_tx_cnt = 0
        zero_val_recv_tx_cnt = 0


        # DEBUG
        if (int(addr_idx) > addr_len):
            print("\n$$$ Finish DEBUG... $$$\n")
            
            saveDataToCsv() 
            return
        
        addr_tx_li = addr_TX_di[addr_idx]
        
        # get number of normal TX
        norm_Tx_count = len(addr_tx_li)
        
        
        # print(f"\n****  DEBUG [{addr_idx}]: <{addr_this}> ****")
        
        # get number of normal tx send from 'addr'
        # get number of normal tx receive from 'addr'
        for tx in addr_tx_li:
            
            
            # OUT (send)
            if (tx['from'] == addr_this):
                out_tx += 1
                
                # get min send amount
                addr_tx_send_di.setdefault(tx['to'], 0)
                addr_tx_send_di[tx['to']] += 1
                
                if (tx['isError'] == "1"):
                    # print("skip...")
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
                
                # : calculate historical balance in Wei first (max or min)
                # val_in_eth = float(tx['value']) / (10 ** 18)
            
                
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
        
        # get gas price (in gwei), fee in AVG, min, MAX
        # extract average, min, Max gas price and gas fee (in ETH)
        gas_price_li =[int(g['gasPrice']) / (10 ** 9) for g in addr_tx_li]
        tx_fee_li =[(int(g['gasPrice']) * float(g['gasUsed'])) / (10 ** 18) for g in addr_tx_li]
        
        
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
        
        
        # 
        
        if (norm_Tx_count == 0):
            out_tx_percent = in_tx_percent = 0
            
        else:
            out_tx_percent = out_tx / (float(norm_Tx_count))
            in_tx_percent = in_tx / (float (norm_Tx_count) )
        
        
        
        # Print out the extracted features

        
        
        # TODO DEBUG
        # # ** no. of send out Txs & receive in Txs
        # print(f"Total no. of Transactions (Normal & Internal): {addr_all_tx_cnt}")
        # print(f"no. of Normal Transactions: {norm_Tx_count}")
        # print(f"no. of Transactions OUT: {out_tx}")
        # print(f"Percentage of Tx OUT = {out_tx_percent}")
        # print(f"no. of Transactions IN: {in_tx}")
        # print(f"Percentage of Tx IN = {in_tx_percent}")
        
        # print(f"\ncurent Balance = {addr_balance}")
        
        # # ** Transaction Value
        # print(f"\nMax send values = {max_val_send}")
        # print(f"min send values = {(min_val_send)}")
        # print(f"Mean send values = {mean_val_send}")
        # print(f"Stdev send values = {(stdev_val_send)}")
        
        # print(f"\nMax recv values = {max_val_recv}")
        # print(f"min recv values = {(min_val_recv)}")
        # print(f"Mean recv values = {mean_val_recv}")
        # print(f"Stdev recv values = {(stdev_val_recv)}")

        # # ** Gas Price
        # print(f"\nAverage Gas Price = {(mean(gas_price_li))}")
        # print(f"Max Gas Price = {max_gas_price}")
        # print(f"min Gas Price = {min_gas_price}")
        # print(f"std Gas Price = {stdev_gas_price}")
        
        
        # # ** Gas Fee
        # print(f"\nAverage Transaction Fee = {mean_transaction_fee}")
        # print(f"Max Transaction Fee = {max_transaction_fee}")
        # print(f"min Transaction Fee = {min_transaction_fee}")
        # print(f"std Transaction Fee = {stdev_transaction_price}")
        
        # # ** unique address Send or receive from
        # print(f"\nTotal no. of uniq address Send to = {uniq_send_address_num}")
        # print(f"Total no. of uniq address Receive from = {uniq_receive_address_num}")
        
        # # ** zero value transaction
        # print(f"\nNumber of Zero value transaction = {zero_val_tx_cnt}")
        # print(f"Number of Zero value SEND transaction = {zero_val_send_tx_cnt}")
        # print(f"Number of Zero value RECEIVE transaction = {zero_val_recv_tx_cnt}")
        
        # # Time Different between Transaction
        # print(f"\nAverage time difference between Transaction = {avg_time_between_tx}")
        # print(f"Average time difference between send Transaction = {avg_time_between_send_tx}")
        # print(f"Average time difference between receive Transaction = {avg_time_between_recv_tx}")
        
        
        
        
        # print("/////////////////\n")
        
        # print(tx_time_li)
        
        addr_features = {
            
            # Basic Attributes
            "index": addr_idx,
            "address": addr_this,
            
            # Transaction Count
            "total_transaction_count": addr_all_tx_cnt,
            "num_of_normal_transaction": norm_Tx_count,
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
            "min_gas_ptice": min_gas_price,
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
        
        All_addr_features_li.append(addr_features)

        

    # print('array:', addr_di.keys())
    
    saveDataToCsv()    
    
    return


def saveDataToCsv():
    print("writing data to csv...")
    make_csv_from_JSON_data(jsonData=All_addr_features_li, csvFilePath="addr_features.csv")
    print("** Data writing Finished **")

if __name__ == "__main__":
    main()


# Example of a Transaction
        # {
        #     "blockHash": "0x29a0c0f7cc25a84f444deccb75c6716310cca4f36ffc9b060445bb256c6ac61c",
        #     "blockNumber": "13793208",
        #     "confirmations": "4641501",
        #     "contractAddress": "",
        #     "cumulativeGasUsed": "4839801",
        #     "from": "0x00009277775ac7d0d59eaad8fee3d10ac6c805e8",
        #     "functionName": "",
        #     "gas": "21000",
        #     "gasPrice": "60313047492",
        #     "gasUsed": "21000",
        #     "hash": "0x9095c86bc7fa3a215da5b0ed5c6c27ee1c9888705aa65ccc14a462432336d01c",
        #     "input": "0x",
        #     "isError": "0",
        #     "methodId": "0x",
        #     "nonce": "753",
        #     "timeStamp": "1639349582",
        #     "to": "0xfe1b6aa4f75ae5475d29e2eaa9e5fe33871834e9",
        #     "transactionIndex": "58",
        #     "txreceipt_status": "1",
        #     "value": "465188923809954681"
        # },