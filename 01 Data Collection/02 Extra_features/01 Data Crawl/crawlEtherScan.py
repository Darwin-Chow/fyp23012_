from urllib.request import Request, urlopen

from requests_html import HTMLSession

from bs4 import BeautifulSoup as soup
import pandas as pd
import requests

import json

from tqdm import tqdm

# 
from threading import Thread

# 
from time import sleep
# from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


from selenium_profiles.profiles import profiles
from selenium.webdriver.common.by import By  # locate elements
from selenium_profiles.webdriver import Chrome
from seleniumwire import webdriver
from selenium_stealth import stealth



# from webdriver_manager.chrome import ChromeDriverManager

from fake_useragent import UserAgent

ua = UserAgent()



import undetected_chromedriver as uc

# 
import sys
sys.path.append('../01 Data Collected/03 utilsFunc')

from utilsFunc import make_json_from_csv, parseJson, getAddrList

# declare
# option = Options()

option = webdriver.ChromeOptions() 

# option = webdriver.ChromeOptions()
# option = Options()
# option = uc.ChromeOptions()
option.binary_location = '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser'

# option.binary_location = '/Users/idea/Downloads/chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'

option.add_argument('--no-sandbox')
option.add_argument('--disable-dev-shm-usage')

# disable image
option.add_argument('--blink-settings=imagesEnabled=false')

# headless
option.add_argument('--headless=new')

profile = profiles.Windows() # or .Android
# profile["proxy"] = {
#     "proxy": "http://134.195.101.34:8080"
# }


# option.add_argument('--headless=new')
headers = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.117 Safari/18614.4.6.1.5'
option.add_argument(f'User-agent={ua.chrome}')

service = Service(executable_path='/Users/idea/Documents/chromedriver_119')

All_data = {}

def main():
    
    args = sys.argv[1:]
    
    num_threads = 4
    start_index = 0
    quick_start_file = None
    
    if (len(args) >= 1):
        num_threads = int(args[0])
    
    if (len(args) >= 2):
        quick_start_file = args[1]
        
        with open (quick_start_file, 'r') as f:
            x = f.read()
            
            quick_data = {int(k): v for (k,v) in json.loads(x).items()}
            
            All_data.update(quick_data)
        
        print("length of quick start:", len(All_data))
        start_index = max(list(map(int, list(All_data.keys()))))
        
        print("start with index (from 1):", start_index)
        
        
        
    print("number of threads:", num_threads)
        
    # 1) get address list to be collected
    addr_list = getAddrList()
    
    NumOfAddr = len(addr_list)
    
    
    # TESTING
    if (len(args) >= 3):
        
        if (args[2] == "all"):
            NumOfAddr = len(addr_list) - start_index + 1
        else:
            NumOfAddr = int(args[2])
    
    if (len(args) >= 4 and args[3] == "fix"):
        start_index = 0
        print("-- Fixing mode is on --\nstart with address no. 1...\n")
    
    
    print("number of address:", NumOfAddr)
    
    prevIdx = start_index
    
    threads = []
    
    drvr_li = []
    
    # final_di = startChromeDriver(addr_li=addr_list)
    
    
    for i in range(num_threads):
        # try selenium driver
        # drvr = webdriver.Chrome(profile, service=service, options = option)
        
        drvr = Chrome(profile, options=option,
                uc_driver=False
                )
        
        drvr.set_window_position(0, 0)

        

        # drvr = Chrome(driver_executable_path='/Users/idea/Documents/chromedriver', options = option)
        drvr_li.append(drvr)
    
    
    # Split the address array into n parts
    for i in range(1, num_threads + 1):
        
        ith_drvr = drvr_li[i - 1]
        
        lenOfEachArr = NumOfAddr // num_threads # e.g., 10 // 4 = 2
        
        i_th_addr_arr = []
        endIdx = 0
        
        if (i != num_threads):
            endIdx = prevIdx + lenOfEachArr
        else:
            endIdx = start_index + NumOfAddr # ** last one
        
        i_th_addr_arr = addr_list[prevIdx: endIdx] # e.g., 1st: [0: 0 + 2 * 1] --> 0, 1
                                                  #       2nd: [2: 2 + 2]     --> 2, 3
                                                  #       3rd: [4: 4 + 2]     --> 4, 5
                                                  #       4th: [6: -1]        --> 6, 7, 8, 9  as it is the last thread
                                                  
        prevIdx = endIdx
        
        # print(f'{i}th array: {[addr["index"] for addr in i_th_addr_arr]}')
        
        
        t = Thread(target=testThreads, args=(i, ith_drvr, i_th_addr_arr))
        
        t.start()
        print(f'Thread {i} started...')
        threads.append(t)
    
    
    # Wait all threads to finish
    for t in threads:
        t.join()
    
    
    print("All threads have finished...")
    
    # sort the data by index    
    index_li = list(map(int, list(All_data.keys())))
    
    # print(f"Results:\n {index_li}")
    print(f"length: {len(All_data)}")
    
    
    minIdx = min(index_li)
    MaxIdx = max(index_li)
    print("min idx:", minIdx)
    print("Max idx:", MaxIdx)
    
    with open(f'addr_max_min_balance_{minIdx}_{MaxIdx}.json', 'w+') as f:
        f.write(json.dumps(All_data, sort_keys = True, indent = 4))

    return
    
    

def startChromeDriver(id, my_drvr, addr_data):

    try:
        
        # for i in tqdm(range(3)):
        retry = None
        
        retryCounter = 0
        
            # addr_data = addr_li[i]
        url = f"https://etherscan.io//address-analytics?m=light&a={addr_data['address']}&lg=en&cc=USD"
            
        my_drvr.get(url)
        sleep(3)
        
        # my_drvr.service.stop()
        
        # sleep(2)
        
        # my_drvr.service.start()
            
                
        while (retry == None):
            if (retryCounter >= 400):
                print("!!! 400 Retry attempted !!!")
                
                if (id == 1):
                    print(f"length: {len(All_data)}")
                    
                    tmp_index_li = list(map(int, list(All_data.keys())))
    
                    tmp_minIdx = min(tmp_index_li)
                    tmp_MaxIdx = max(tmp_index_li)
                    print("min idx:", tmp_minIdx)
                    print("Max idx:", tmp_MaxIdx)
                    
                    with open(f'ERR_addr_max_min_balance_{tmp_minIdx}_{tmp_MaxIdx}_ERR.json', 'w+') as f:
                        f.write(json.dumps(All_data, sort_keys = True, indent = 4))
                    
                    
                    print("tmp JSON file saved...")
                
                raise Exception("!!! Timeout Error!!")
                
            html = my_drvr.page_source
            page_soup = soup(html, "html.parser")
                    
            retry = page_soup.find(id='high_bal_eth_value')
                    
                # print("retrying...")
            sleep(0.5)
                
            retryCounter += 1

        # features to extracted
        highestEth = page_soup.find(id='high_bal_eth_value').get_text()
        highestEth_date = page_soup.find(id='high_bal_eth_date').get_text()[3:]
                
        lowestEth = page_soup.find(id="low_bal_eth_value").get_text()
        lowestEth_date = page_soup.find(id='low_bal_eth_date').get_text()[3:]

        # print(f"\n Addr: <{addr_data['address']}>")
        # print(f"\n-- Highest Balance (in ETH):\t{highestEth.split(' ETH')[0]}\t({highestEth_date}) --")
                
        # print(f"-- lowest Balance (in ETH):\t{lowestEth.split(' ETH')[0]}\t({lowestEth_date}) --\n")

            
        # save the data
        addr_new_features = {
                'index': addr_data['index'],
                'address': addr_data['address'],
                'flag': addr_data['flag'],
                'highestBalance': highestEth.split(' ETH')[0],
                'highestBalanceDate': highestEth_date,
                'lowestBalance': lowestEth.split(' ETH')[0],
                'lowestBalance_date': lowestEth_date
        }
            
        data = addr_new_features
            
        # print(data)

        return data
        
    except Exception as e:
        print(e)
        my_drvr.quit()

    # finally:
    #         # pass
    #     my_drvr.quit()


def testThreads(id, mydrvr, addr_li: []):
    # print(f'array: {[addr["index"] for addr in addr_li]}')
    
    # addrs = [(addr["address"], addr["index"]) for addr in addr_li]
    
    all_keys = All_data.keys()
    
    for addr_data in tqdm(addr_li):
        
        acc_idx = int(addr_data["index"])
        
        if (( acc_idx not in all_keys) or ( (acc_idx in all_keys ) and All_data[acc_idx] is None)):
        
            # print(f"starting with idx {acc_idx}...")
            d = startChromeDriver(id=id, my_drvr=mydrvr, addr_data=addr_data)
            # print(f"Tx: <{acc_addr}>:", len(d))

            All_data[acc_idx] = d


def interceptor(request):
    request.headers['New-Header'] = 'Some Value'     

if __name__ == "__main__":
    # Usage: python3 crawlEtherScan.py <num_of_threads> <quick_start_file_path>
    main()
    
    
    
    
    
    
    # headers = {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'} 


    # Request
    # html = requests.get(url,headers=headers)