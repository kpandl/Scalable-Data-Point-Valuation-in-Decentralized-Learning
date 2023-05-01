import ipfsapi
import os
#from dotenv import load_dotenv
from web3 import Web3
import json
from web3.middleware import geth_poa_middleware

orchestrator_path = os.path.join(os.getcwd(), "orchestrator.json")
f = open(orchestrator_path)
orchestrator_data = json.load(f)
f.close()

network_apis_ath = os.path.join(os.getcwd(), "network_apis.json")
f = open(network_apis_ath)
network_apis_data = json.load(f)
f.close()

my_contract_bytecode = network_apis_data["private_blockchain"]["smart_contract"]["bytecode"]
my_contract_ABI = network_apis_data["private_blockchain"]["smart_contract"]["ABI"]

api_url = network_apis_data["private_blockchain"]["url"]
private_key = orchestrator_data["private_blockchain"]["private_key"]
from_account = orchestrator_data["private_blockchain"]["address"]

ipfs_api_instance = ipfsapi.Client('127.0.0.1', 5001)    

def ipfs_add (filepath):
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

file_path = os.path.join(os.getcwd(), "model_initial.pt")
ipfs_hash, filename = ipfs_add(file_path)

web3 = Web3(Web3.HTTPProvider(api_url))
web3.middleware_onion.inject(geth_poa_middleware, layer=0)
nonce = web3.eth.get_transaction_count(from_account)

contract = web3.eth.contract(
    abi=my_contract_ABI,
    bytecode=my_contract_bytecode
)

encoded_ipfs_hash = encode_hex(ipfs_hash)

tx = contract.constructor(network_apis_data["private_blockchain"]["contract_parameters"]["institution_addresses"], encoded_ipfs_hash).build_transaction({
    'nonce': nonce
})

gas = web3.eth.estimate_gas(tx)
tx['gas'] = int(2 * gas)
print("Gas: " + str(gas))
signed_tx = web3.eth.account.sign_transaction(tx, private_key)
tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
print("Transaction hash: " + str(web3.to_hex(tx_hash)))

# wait for the transaction to be mined
web3.eth.wait_for_transaction_receipt(tx_hash)

# sleep for 10 seconds to make sure the contract is mined

contract_address = web3.eth.get_transaction_receipt(tx_hash)['contractAddress']

# store the contract address in a file
path = os.path.join(os.getcwd(), "private_contract_address.txt")
# if file exists, delete it

if os.path.exists(path):
  os.remove(path)

with open(path, 'w') as f:
    f.write(contract_address)