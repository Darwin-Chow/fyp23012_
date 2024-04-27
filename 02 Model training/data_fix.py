

import re

def main():
    with open("combined_address_features_complete.csv", 'r') as f:
        l = f.read().split("\n")
    
    newFile_line = []

    for ll in l:
        
        # if (ll != l[1504]):
        #     continue
        
        if ("\"" in ll):
            # print(ll, "\n")
            
            quoted_str = re.findall('"([^"]*)"', ll)
            
            # print(quoted_str)
            
            for q in (quoted_str):
                if q in ll:
                    # tmp = q.replace("\"", "")
                    tmp = q.replace(",", "")
                    
                    # remove comma
                    ll = ll.replace(q, tmp)
                    
                    # remove quotes
                    ll = ll.replace("\"", "")
                    # print("new ll:\n", ll)
            
            newFile_line.append(ll)
            
        else:
            newFile_line.append(ll)
            # start_idx = ll.find("\"")
            
            # if (start_idx != -1):
            #     for i in range(start_idx, len(ll)):
            #         ch = ll[i]
            #         if (ch != "\""):
            
            # return
    with open("combined_fix_bug.csv", "w+") as ot:
        for line in newFile_line:
            ot.write(line + "\n")


if __name__ == "__main__":
    main()