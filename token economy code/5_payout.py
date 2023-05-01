# retrieve original model hash from smart contract

import ipfsapi
import os
#from dotenv import load_dotenv
from web3 import Web3
import json
from web3.middleware import geth_poa_middleware
import time
import pandas as pd
from helper import ipfs_add, encode_hex, save_from_ipfs

orchestrator_path = os.path.join(os.getcwd(), "orchestrator.json")
f = open(orchestrator_path)
orchestrator_data = json.load(f)
f.close()

network_apis_ath = os.path.join(os.getcwd(), "network_apis.json")
f = open(network_apis_ath)
network_apis_data = json.load(f)
f.close()


public_contract_bytecode = network_apis_data["public_blockchain"]["smart_contract"]["bytecode"]
public_contract_ABI = network_apis_data["public_blockchain"]["smart_contract"]["ABI"]


public_api_url = network_apis_data["public_blockchain"]["url"]
public_private_key = orchestrator_data["public_blockchain"]["private_key"]
public_from_account = orchestrator_data["public_blockchain"]["address"]


private_contract_txt_path = os.path.join(os.getcwd(), "private_contract_address.txt")
f = open(private_contract_txt_path)
private_contract_address = f.read()
f.close()

public_contract_txt_path = os.path.join(os.getcwd(), "public_contract_address.txt")
f = open(public_contract_txt_path)
public_contract_address = f.read()
f.close()

ipfs_api_instance = ipfsapi.Client('127.0.0.1', 5001)

public_web3 = Web3(Web3.HTTPProvider(public_api_url))
public_web3.middleware_onion.inject(geth_poa_middleware, layer=0)
public_nonce = public_web3.eth.get_transaction_count(public_from_account)

public_contract = public_web3.eth.contract(
    address=public_contract_address,
    abi=public_contract_ABI,
    bytecode=public_contract_bytecode
)

# payout
tx = public_contract.functions.payBalanceToAllHospitalsImmediately().build_transaction({
'nonce': public_nonce,
'from': public_from_account
})

gas = public_web3.eth.estimate_gas(tx)
tx['gas'] = int(1 * gas)
print("Gas: " + str(gas))
signed_tx = public_web3.eth.account.sign_transaction(tx, public_private_key)
tx_hash = public_web3.eth.send_raw_transaction(signed_tx.rawTransaction)
print("Transaction hash: " + str(public_web3.to_hex(tx_hash)))
public_web3.eth.wait_for_transaction_receipt(tx_hash)

# sleep for 10 seconds to make sure the transaction is mined
time.sleep(10)

transaction_receipt = public_web3.eth.get_transaction_receipt(tx_hash)
transaction_cost = transaction_receipt['gasUsed'] * public_web3.eth.get_transaction(tx_hash)['gasPrice'] / 10**18

print("Transaction cost:", transaction_cost, "gas used:", str(transaction_receipt['gasUsed']))

# store the transaction cost in a csv file

with open('public_transaction_cost_orchestrator.csv', 'a') as f:
    f.write("contract_payout,"+str(transaction_cost)+","+str(transaction_receipt['gasUsed'])+"\n")

public_nonce += 1


print("finished")