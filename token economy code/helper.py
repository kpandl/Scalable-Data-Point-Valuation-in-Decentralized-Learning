import json
import os
import requests

def ipfs_add (filepath, ipfs_api_instance):
    print(f"INFO - Uploading {filepath} to IPFS network")
    #match is necessary to only upload one file, not the directory
    print(f"match: {os.path.basename(filepath)}")
    print
    res = ipfs_api_instance.add(filepath)
    #add api weirdly also adds directories with a seperate hash, catching this here:
    if type(res) is not dict:
        res = res[0]
    print(f"INFO - Successful. IPFS hash: {res['Hash']}")
    return res['Hash'], res['Name']

def encode_hex(hash):    
    dict = {"ipfs-hash": hash}
    j=json.dumps(dict)
    output = j.encode("utf-8").hex()  
    return "0x87654321" + output

# download model from ipfs
def save_from_ipfs(hash, file, foldername=None):
    url = "http://127.0.0.1:5001/api/v0/cat?arg="+hash
    res = requests.post(url) 
    print(f"INFO - Writing file to {file}")
    if(foldername == None):
        path = os.path.join(os.getcwd(), file)
    else:
        path = os.path.join(os.getcwd(), foldername, file)
    with open(path, 'wb') as fout:
        fout.write(res.content)