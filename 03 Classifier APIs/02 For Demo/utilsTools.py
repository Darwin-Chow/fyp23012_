
from enum import Enum

# Global variables

# create ENUM for mutiple class of Acc Address
# 1) scam address
# 2) normal address
# 3) zero-transaction address
class ETH_accounts(Enum):
    ILLICIT = "0x0a52ecaa61268c6a5cf9cd6b1378531a4672601b"  # BitKeep Attacker
    ILLICIT2 = "0x80d7bb18521acbef32d7906502ebe94928690e93"
    
    NORMAL = "0xd0539E3f72744ed74C1Bdc6ce01beEB3aD61Ca31"   # Random Address
    NORMAL2 = "0x00000000000000ADc04C56Bf30aC9d3c0aAF14dC"
    
    ZEROTX = "0x00c8bd3CD1e649A3Fd2A89b3edC1C2aB631227a0"   # index 8151 from prev data
    
    TEST = "0x00062d1dd1afb6fb02540ddad9cdebfe568e0d89"     # index 5 --> Many TX
    DARKWEB = "0xf27EAe399f186600Dc6e5A418793C4A3D58a74e7"  # An Address from dark web





class ExplorerAPI:
    def __init__(self, URL: str, key_li: []) -> None:
        self.URL = URL
        self.key_li = key_li

# -------- OKLink explorer -------

Oklink_API_Endpoint = "https://www.oklink.com"
Oklink_API_URL = f"{Oklink_API_Endpoint}/api/v5/explorer/address/address-summary"
Oklink_API_Key = "1a4ce7da-22ef-49af-88ae-e8c5ff75adb5"

Oklink_API = ExplorerAPI(URL = Oklink_API_URL, 
                         key_li = [Oklink_API_Key])



# -------------------------------

# -------- Etherscan.io --------
Etherscan_API_URL = "https://api.etherscan.io/api"
Etherscan_API_KEY_LIST = ["UF99YFKUDW52Y6TU9I9C2JIBIXP5P6BN6F",
                "BF562BUQGHGT883SAV18VZ18AKH7PTDKSY",
                "D1UIKNGH3Z3QA8AVFJ1BJHEMD3SR35K5TM"]

Etherscan_API = ExplorerAPI(URL = Etherscan_API_URL, 
                            key_li = Etherscan_API_KEY_LIST)



# -------------------------------

