
import csv
import json
import sys

import traceback

import datetime
import time
import calendar

from tqdm import tqdm


def main():
    
    args = sys.argv[1:]
    
    input_file_path = args[0]
    
    csvFormat = True if (".json" in input_file_path) else False
    
    output_file_path = f"./Converted_file/{input_file_path.split('.')[-2].split('/')[-1]}_converted"
    
    if (csvFormat):
        output_file_path += ".csv"
    else:
        output_file_path += ".json"
    
    if (len(args) > 1):
        output_file_path = args[1]
    
    print(f"\n***\n Output to \n{output_file_path}\n***\n")
    
    try:
        if (".json" in input_file_path):
            
            # make isList to False to parse nested dict
            make_csv_from_json_file(jsonFilePath=input_file_path, csvFilePath=output_file_path, isList=False)
            
            print(f"output file is located at {output_file_path}")
            
        
        elif (".csv" in input_file_path):
            make_json_from_csv(csvFilePath=input_file_path, jsonFilePath=output_file_path)
            
            print(f"output file is located at {output_file_path}")
            
        else:
            print("!! Error !!")
        
        
        
    
    except Exception as e:
        print(e, "\n")
        traceback.print_exc() 
        print()
    
    return

def getAddrList() -> []:
    addrLi = []
    
    with open('/Users/idea/Documents/FYP/01 Data Collected/03 utilsFunc/addr-list.json', 'r') as f:
        x = f.read()
        addrLi = json.loads(x)
    
    
    return addrLi
    

def parseJson(filePath: str, isQuoted: bool = False) -> ([], int):
    addr_sum_list = []
    with open(filePath, 'r') as jsonfile:
        x = jsonfile.read()
        # print(x)
        if (isQuoted):
            addr_sum_list = json.loads(json.loads(x))
        else:
            addr_sum_list = json.loads(x)

    
    return addr_sum_list, len(addr_sum_list)


# Function to convert a CSV to JSON
# Takes the file paths as arguments
def make_json_from_csv(csvFilePath, jsonFilePath):
     
    # create a dictionary
    data = []
    
    
    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)

        # Convert each row into a dictionary 
        # and add it to data
        for rows in csvReader:
            data.append(rows)
 
                # Assuming a column named 'No' to
                # be the primary key
                # key = rows['No']
                # data[key] = rows
                
    with open(jsonFilePath, 'x', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, sort_keys = False, indent = 4)) #for pretty
 
 
def make_csv_from_json_file(jsonFilePath, csvFilePath, isList = True):
    
    jsonData = None
    
    jsonDataLi, _ = parseJson(jsonFilePath)
    
    if (not isList):
        jsonData = list(jsonDataLi.values())
    else:
        jsonData = jsonDataLi
    
    
    print("\nFirst Data:\n", jsonData[0], "\n")
    
    try:
        # [Specical Case] for datetime only
        
        # data = jsonData[0]
        for data in jsonData:
            
            if (data['highestBalanceDate'] == ""):
                continue
            
            dt_hi = datetime.datetime.strptime(data['highestBalanceDate'].replace("Sept", "Sep"), "%a %d, %b %Y")
            dt_hi_tmp = calendar.timegm(dt_hi.utctimetuple())
            data['highestBalanceDate'] = dt_hi_tmp * 1000
            
            dt_low = datetime.datetime.strptime(data['lowestBalance_date'].replace("Sept", "Sep"), "%a %d, %b %Y")
            dt_low_tmp = calendar.timegm(dt_low.utctimetuple())
            data['lowestBalance_date'] = dt_low_tmp * 1000
        
        # test_li = data['highestBalanceDate'].split(" ")
        
        # # test_li[0]
        
        # a_num = test_li[0]
        # d_num = test_li[1][0:-1] .rjust(2, '0')
        # b_num = test_li[2]
        # Y_num = test_li[3]
        
        # my_date = f"{a_num} {d_num} {b_num} {Y_num}"
            
        # print("my-date:", data['highestBalanceDate'])
        
    except Exception as e:
        print("ERROR: ", e)
        pass
    
    # 
    
    # Testing
    make_csv_from_JSON_data(jsonData[0], csvFilePath)
    
    # make_csv_from_JSON_data(jsonData, csvFilePath)


def make_csv_from_JSON_mutl_data(jsonDataList, csvFilePath):
    
    with open(csvFilePath, 'x') as csvF:
        csv_writer = csv.writer(csvF)
        
        cur_id = 0
        
        for JsonData in tqdm(jsonDataList):
        
            count = 0
            for data in JsonData:
                if count == 0:
                    # header = data.keys()
                    csv_writer.writerow(cur_id)
                    count += 1
                csv_writer.writerow(data.values())
            cur_id += 1
    
    
 
def make_csv_from_JSON_data(jsonData, csvFilePath):
    with open(csvFilePath, 'x') as csvF:
        csv_writer = csv.writer(csvF)
        count = 0
        for data in jsonData:
            if count == 0:
                header = data.keys()
                csv_writer.writerow(header)
                count += 1
            csv_writer.writerow(data.values())



if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Usage: python3 utilsFunc <input_file_path> <output_file_path (optional)>")
    
    elif ((".csv" not in sys.argv[1]) and ( ".json" not in sys.argv[1])):
        print(f"Invalid file format:\n {sys.argv[1]}\n '*.csv' or '*.json' file expected")
    
    else:
        main()