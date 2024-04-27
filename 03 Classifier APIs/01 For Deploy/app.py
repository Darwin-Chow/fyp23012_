from flask import Flask


import json
from web3 import Web3

from test_collect_real_data import classify

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/address/<addr>', methods=['GET'])
def classifyAccount(addr):
    
    res = {
        'code': '',
        'data': '',
        'msg': ''
    }
    
    if (not Web3.is_address(addr)):
        res['code'] = '404'
        res['msg'] = 'Invalid Address'
        return json.dumps(res, sort_keys = False, indent = 4)
    
    acc_features, probAsIllicit = classify(addr)
    # acc_features['probability (illicit)'] = [probAsIllicit]
    
    res['code'] = 200
    res['data'] = acc_features
    res['probability (illicit)'] = [probAsIllicit]
    
    
    print(json.dumps(res, sort_keys = False, indent = 4))
    print(f"Probability (Illicit): {probAsIllicit}%")
    return json.dumps(res, sort_keys = False, indent = 4)



