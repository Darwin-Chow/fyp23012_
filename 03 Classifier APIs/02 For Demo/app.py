from flask import Flask

# For CORS (Cross Origin Resource Sharing)
from flask_cors import CORS, cross_origin


import json
from web3 import Web3

from test_collect_real_data import classify

from visual_data import retreiveFirstOrderTx, retrieveMoneyFlowTX

app = Flask(__name__)

# CORS
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



@cross_origin()
@app.route("/")
def hello():
    return "Hello, World!"


@cross_origin()
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
    
    if (probAsIllicit == "NaN"):
        res['msg'] = "Zero Transaction Address"
    
    
    print(json.dumps(res, sort_keys = False, indent = 4))
    print(f"Probability (Illicit): {probAsIllicit}%")
    return json.dumps(res, sort_keys = False, indent = 4)


@cross_origin()
@app.route('/transaction/<addr>', methods=['GET'])
def retrieveTransaction(addr):
    
    # Data Format
    res = {
        'code': '',
        'data': '',
        'msg': ''
    }
    
    # Filter Invalid Format
    if (not Web3.is_address(addr)):
        res['code'] = '404'
        res['msg'] = 'Invalid Address'
        return json.dumps(res, sort_keys = False, indent = 4)
    
    
    # 1st Order Transaction ONLY
    # (norm_addr_tx_li, _, _, _) = retreiveFirstOrderTx(addr)
    
    # Retrieve Transaction at least 2nd Order
    norm_addr_tx_li = retrieveMoneyFlowTX(addr)
    
    res['code'] = 200
    res['data'] = norm_addr_tx_li
    
    print(json.dumps(res, sort_keys = False, indent = 4))
    return json.dumps(res, sort_keys = False, indent = 4)
    
    
    
    
    

