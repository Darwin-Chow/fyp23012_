import pandas as pd 
  
# reading csv files 

# first 3 merging
# data1 = pd.read_csv('Data/address_summary_complete.csv') 
# data2 = pd.read_csv('Data/addr_features.csv') 
# data3 = pd.read_csv('Data/addr_max_min_balance_1_13941_converted.csv') 

data_merge = pd.read_csv('Data/addr_internal_features.csv') 
data_int = pd.read_csv('Data/norm_address_features_complete.csv') 


# using merge function by setting how='outer' 
# data1_2 = pd.merge(data1, data2,  
#                    on='address',  
#                    how='left') 


# combined_df = pd.merge(data1_2, data3,  
#                    on='address',  
#                    how='left') 

combined_df = pd.merge(data_int,  data_merge,
                   on='address',  
                   how='outer') 


# displaying result
print(combined_df)

combined_df.to_csv('combined_address_features_complete.csv', index=False)


