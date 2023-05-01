import os
#from dotenv import load_dotenv
from web3 import Web3
import json
import time

orchestrator_path = os.path.join(os.getcwd(), "orchestrator.json")
f = open(orchestrator_path)
orchestrator_data = json.load(f)
f.close()

network_apis_ath = os.path.join(os.getcwd(), "network_apis.json")
f = open(network_apis_ath)
network_apis_data = json.load(f)
f.close()

EIP20_ABI = network_apis_data["public_blockchain"]["stable_coin_ABI"]
stable_coin_address = network_apis_data["public_blockchain"]["contract_parameters"]["stablecoin_contract_address"]
my_contract_bytecode = network_apis_data["public_blockchain"]["smart_contract"]["bytecode"]
my_contract_ABI = network_apis_data["public_blockchain"]["smart_contract"]["ABI"]

api_url = network_apis_data["public_blockchain"]["url"]
private_key = orchestrator_data["public_blockchain"]["private_key"]
from_account = orchestrator_data["public_blockchain"]["address"]

public_contract_txt_path = os.path.join(os.getcwd(), "public_contract_address.txt")
f = open(public_contract_txt_path)
public_contract_address = f.read()
f.close()

web3 = Web3(Web3.HTTPProvider(api_url))
nonce = web3.eth.get_transaction_count(from_account)

contract = web3.eth.contract(
    abi=my_contract_ABI,
    bytecode=my_contract_bytecode
)

usdc = web3.eth.contract(address=stable_coin_address, abi=EIP20_ABI)

tx = usdc.functions.transfer(public_contract_address, 100000000000000000).build_transaction({
   'from': from_account, 
   'nonce': nonce
   })

gas = web3.eth.estimate_gas(tx)
tx['gas'] = 1 * gas
print("Gas: " + str(gas))
signed_tx = web3.eth.account.sign_transaction(tx, private_key)
tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
print("Transaction hash: " + str(web3.to_hex(tx_hash)))

# wait for the transaction to be mined
web3.eth.wait_for_transaction_receipt(tx_hash)

# sleep for 10 seconds to make sure the transaction is mined
time.sleep(10)

transaction_receipt = web3.eth.get_transaction_receipt(tx_hash)
transaction_cost = transaction_receipt['gasUsed'] * web3.eth.get_transaction(tx_hash)['gasPrice'] / 10**18

print("Transaction cost:", transaction_cost)

# store the transaction cost in a csv file

with open('public_transaction_cost_orchestrator.csv', 'a') as f:
    f.write("contract_funding,"+str(transaction_cost)+","+str(transaction_receipt['gasUsed'])+"\n")

print("Done!")