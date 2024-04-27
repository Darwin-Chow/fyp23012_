
# HTTP request & file
import requests
# import sys
import json
from time import sleep

# extract Features
import numpy as np


from utilsTools import ETH_accounts, Etherscan_API, Oklink_API, ExplorerAPI

from test_collect_real_data import retrieveAccTx

def retrieveMoneyFlowTX(addr):
    first_order_tx, addr_li, min_block, max_block = retreiveFirstOrderTx(addr)
    
    first_second_order_tx = retreiveTxFromAddrLi(addr_li=addr_li, addr_tx=first_order_tx, startBlock=min_block, endBlock=max_block, n=20)
    
    print(f"\n*** 1st + 2nd Order Length {len(first_second_order_tx)} ***\n")
    
    # first_sec_third_order_tx =
    
    return first_second_order_tx
    
    



def retreiveFirstOrderTx(addr):
    norm_tx_li = retrieveAccTx(addr, isNormal=True)
    
    (norm_tx_li_20, addr_20, min_block, max_block) = get_n_tx_addr_from_tx(queryAddr = addr, tx_li = norm_tx_li)
    
    return (norm_tx_li_20, addr_20, min_block, max_block)


def get_n_tx_addr_from_tx(queryAddr, tx_li: [], n = 20):
    res = []
    M = {}
    
    min_block = 99999999
    max_block = -1
    
    
    for tx in tx_li:
        
        addr = tx['from'] if (tx['from'] != queryAddr) else tx['to']
        
        if (addr == queryAddr):
            continue
        
        if ( not (addr in M) ):
            M[addr] = []
        
        M[addr].append(tx)
        
        # Get Block Range
        blockNum = int(tx['blockNumber'])
        
        if (blockNum < min_block):
            min_block = int(tx['blockNumber'])
         
        if (blockNum > max_block):
            max_block = int(tx['blockNumber'])
        
        
        if (len(list(M.keys())) >= n):
            break
    
    
    for addr_txs in list(M.values()):
        res.extend(addr_txs)
    
    print("\n*** Keys: ", list(M.keys()), "***\n")
    print("\n*** TX len: ", len(res), "***\n")
    print(f"\n*** Block Range: {min_block} to {max_block} ***\n")
    
    
    
    addr_20 = list(M.keys())
    
    
    
    
    return (res, addr_20, min_block, max_block)


def retreiveTxFromAddrLi(addr_li: [], addr_tx: [], startBlock, endBlock, n = 50):
    
    addrs_tx_li = []
    addrs_tx_li.extend(addr_tx)
    
    for addr in addr_li:
        tx_li = retrieveAccTx(addr, isNormal=True, startBlock=startBlock, endBlock=endBlock, offset=n)
        
        addrs_tx_li.extend(tx_li)
    
    
    return addrs_tx_li


def retreiveEach20TxFromAddrLi(addr_li: [], addr_tx: [], startBlock, endBlock, n = 20):
    
    addrs_tx_li = []
    addrs_tx_li.extend(addr_tx)
    
    for addr in addr_li:
        
        # get account tx for each addre
        tx_li = retrieveAccTx(addr, isNormal=True, startBlock=startBlock, endBlock=endBlock, offset=n)
        
        # get the latest 10 address's tx
        (norm_tx_li_10, addr_10, min_block, max_block) = get_n_tx_addr_from_tx(queryAddr = addr, tx_li = tx_li, n=10)

        
        addrs_tx_li.extend(tx_li)
    
    
    return addrs_tx_li