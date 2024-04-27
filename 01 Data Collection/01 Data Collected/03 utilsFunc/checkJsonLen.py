
import csv
import json
import sys


def parseJson(filePath, isQuoted = False):
    addr_sum_list = []
    with open(filePath, 'r') as jsonfile:
        x = jsonfile.read()
        # print(x)
        if (isQuoted):
            addr_sum_list = json.loads(json.loads(x))
        else:
            addr_sum_list = json.loads(x)

    
    return addr_sum_list, len(addr_sum_list)

def main():
    
    args = sys.argv[1:]
    
    filePath = args[0]
    
    options = False
    
    if (len(args) >= 2):
        options = True if (args[1]) else False
    
    
    jsonArr, jsonLen = parseJson(filePath, options)
    
    print("1st JSON item: \n", jsonArr[0])
    
    print("Length of JSON array:", jsonLen)
    
    return


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Usage: python3 checkJsonLen.py <file_path> [0: JSON without quote (default) | 1 : JSON with quote]")
    else:
        main()
    
    
    


