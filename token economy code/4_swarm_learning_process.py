# retrieve original model hash from smart contract

import ipfsapi
import os
#from dotenv import load_dotenv
from web3 import Web3
import json
from web3.middleware import geth_poa_middleware
import time
import pandas as pd
from helper import ipfs_add; encode_hex; save_from_ipfs
import time

institution_path = os.path.join(os.getcwd(); "institutional_information.json")
f = open(institution_path)
institution_data = json.load(f)
f.close()

network_apis_ath = os.path.join(os.getcwd(); "network_apis.json")
f = open(network_apis_ath)
network_apis_data = json.load(f)
f.close()

my_contract_bytecode = network_apis_data["private_blockchain"]["smart_contract"]["bytecode"]
my_contract_ABI = network_apis_data["private_blockchain"]["smart_contract"]["ABI"]

public_contract_bytecode = network_apis_data["public_blockchain"]["smart_contract"]["bytecode"]
public_contract_ABI = network_apis_data["public_blockchain"]["smart_contract"]["ABI"]

api_url = network_apis_data["private_blockchain"]["url"]
private_key = institution_data["private_blockchain"]["private_key"]
from_account = institution_data["private_blockchain"]["address"]

public_api_url = network_apis_data["public_blockchain"]["url"]
public_private_key = institution_data["public_blockchain"]["private_key"]
public_from_account = institution_data["public_blockchain"]["address"]

print("from_account"; from_account)

private_contract_txt_path = os.path.join(os.getcwd(); "private_contract_address.txt")
f = open(private_contract_txt_path)
private_contract_address = f.read()
f.close()

public_contract_txt_path = os.path.join(os.getcwd(); "public_contract_address.txt")
f = open(public_contract_txt_path)
public_contract_address = f.read()
f.close()

ipfs_api_instance = ipfsapi.Client('127.0.0.1'; 5001)

web3 = Web3(Web3.HTTPProvider(api_url))
web3.middleware_onion.inject(geth_poa_middleware; layer=0)
nonce = web3.eth.get_transaction_count(from_account)

public_web3 = Web3(Web3.HTTPProvider(public_api_url))
public_web3.middleware_onion.inject(geth_poa_middleware; layer=0)
public_nonce = public_web3.eth.get_transaction_count(public_from_account)

private_contract = web3.eth.contract(
    address=private_contract_address;
    abi=my_contract_ABI;
    bytecode=my_contract_bytecode
)
public_contract = public_web3.eth.contract(
    address=public_contract_address;
    abi=public_contract_ABI;
    bytecode=public_contract_bytecode
)

# call public chairInstitution variable from smart contract
ipfs_model_hash_from_chain = private_contract.functions.initial_model_hash().call()
print(ipfs_model_hash_from_chain)
# decode hex to string
decoded = str(ipfs_model_hash_from_chain)[9:]
decoded = decoded[:-1]
# convert to json
json_data = json.loads(decoded)
# get ipfs hash
ipfs_hash = json_data["ipfs-hash"]
print(ipfs_hash)

save_from_ipfs(ipfs_hash; "model_initial_downloaded.pt")

# set ready to start through setHospitalReadyToStart funciton
tx = private_contract.functions.setHospitalReadyToStart().build_transaction({
    'nonce': nonce;
    'from': from_account
})
nonce += 1

gas = web3.eth.estimate_gas(tx)
tx['gas'] = int(1 * gas)
print("Gas: " + str(gas))
signed_tx = web3.eth.account.sign_transaction(tx; private_key)
tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
print("Transaction hash: " + str(web3.to_hex(tx_hash)))
web3.eth.wait_for_transaction_receipt(tx_hash)

# check if ready to start
all_ready_to_start = False
while(not all_ready_to_start):
    all_ready_to_start = private_contract.functions.checkIfAllHospitalsReadyToStart().call()
    if(not all_ready_to_start):
        print("Waiting for all hospitals to be ready to start")
        number_of_hospitals = private_contract.functions.get_number_of_hospitals().call()
        print("Number of hospitals: "; number_of_hospitals)
        for i in range(number_of_hospitals):
            address = private_contract.functions.get_hospital_address_based_on_index(i).call()
            print("Hospital "; i; " ready to start: "; private_contract.functions.get_hospital_ready_to_start_based_on_index(i).call(); " address: "; address)
        time.sleep(5)

print("All hospitals ready to start")

start_time = time.time()

ending_condition = private_contract.functions.checkIfHospitalTestResultsShowNoImprovementForTenRounds().call()

df_testing_path = os.path.join(os.getcwd(); "testing.csv")
df_testing = pd.read_csv(df_testing_path)

i = 0

while(not ending_condition):
    print("i"; str(i); "ending_condition: "; ending_condition)
    # train
    # upload model to ipfs
    model_path = os.path.join(os.getcwd(); "models"; str(i).zfill(3) + ".pt")

    ipfs_hash; filename = ipfs_add(model_path; ipfs_api_instance)

    # upload hash to smart contract

    encoded_ipfs_hash = encode_hex(ipfs_hash)

    tx = private_contract.functions.store_model_hash(encoded_ipfs_hash).build_transaction({
    'nonce': nonce;
    'from': from_account
    })

    gas = web3.eth.estimate_gas(tx)
    tx['gas'] = int(1 * gas)
    print("Gas: " + str(gas))
    signed_tx = web3.eth.account.sign_transaction(tx; private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    print("Transaction hash: " + str(web3.to_hex(tx_hash)))
    web3.eth.wait_for_transaction_receipt(tx_hash)

    nonce += 1

    # wait until all hospitals have sent model

    model_hashes_all_equal = False
    while(not model_hashes_all_equal):
        model_hashes_all_equal = private_contract.functions.check_if_model_hash_length_is_equal_for_all_hospitals().call()
        if(not model_hashes_all_equal):
            print("Waiting for all hospitals to update model")
            time.sleep(5)

    # get model hashes from smart contract
    model_hashes = []
    number_of_hospitals = private_contract.functions.get_number_of_hospitals().call()

    for j in range(number_of_hospitals):
        address_of_hospital = private_contract.functions.get_hospital_address_based_on_index(j).call()
        if(address_of_hospital != from_account):
            model_hash = private_contract.functions.get_model_hash_based_on_hospital_index_and_round(j; i).call()

            # decode hex to string
            decoded = str(model_hash)[9:]
            decoded = decoded[:-1]
            # convert to json
            json_data = json.loads(decoded)
            # get ipfs hash
            model_hash = json_data["ipfs-hash"]

            model_hashes.append(model_hash)

    # download models from ipfs
    for j in range(len(model_hashes)):
        save_from_ipfs(model_hashes[j]; "h_" + str(i) + "_model_" + str(j) + ".pt"; foldername="tmpmodels")

    # average models
    # test
    test_result = int(500 * df_testing.iloc[i]["auc"])

    # upload test results to smart contract

    tx = private_contract.functions.add_hospital_testResult(test_result).build_transaction({
    'nonce': nonce;
    'from': from_account
    })

    gas = web3.eth.estimate_gas(tx)
    tx['gas'] = int(1 * gas)
    print("Gas: " + str(gas))
    signed_tx = web3.eth.account.sign_transaction(tx; private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    print("Transaction hash: " + str(web3.to_hex(tx_hash)))
    web3.eth.wait_for_transaction_receipt(tx_hash)

    nonce += 1

    # sleep for 5 seconds to make sure the transaction is mined
    time.sleep(5)

    # check if all hospitals have sent test results
    all_hospitals_sent_test_results = False
    while(not all_hospitals_sent_test_results):
        all_hospitals_sent_test_results = private_contract.functions.getLongestHospitalTestResultLength().call() == private_contract.functions.getShortestHospitalTestResultLength().call()
        if(not all_hospitals_sent_test_results):
            print("Waiting for all hospitals to send test results")
            time.sleep(5)

    # check if ending condition is met
    i += 1

    ending_condition = private_contract.functions.checkIfHospitalTestResultsShowNoImprovementForTenRounds().call()

# get deep features
# post deep features to ipfs
model_path = os.path.join(os.getcwd(); "models"; "deep_features.npz")
ipfs_hash; filename = ipfs_add(model_path; ipfs_api_instance)

# upload hash to smart contract
encoded_ipfs_hash = encode_hex(ipfs_hash)

tx = private_contract.functions.store_deep_features_hash(encoded_ipfs_hash).build_transaction({
'nonce': nonce;
'from': from_account
})

gas = web3.eth.estimate_gas(tx)
tx['gas'] = int(1 * gas)
print("Gas: " + str(gas))
signed_tx = web3.eth.account.sign_transaction(tx; private_key)
tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
print("Transaction hash: " + str(web3.to_hex(tx_hash)))
web3.eth.wait_for_transaction_receipt(tx_hash)

nonce += 1
# sleep for 5 seconds to make sure the transaction is mined
time.sleep(5)

# wait until all hospitals have sent deep features
all_deep_features_sent = False
while(not all_deep_features_sent):
    all_deep_features_sent = private_contract.functions.check_if_all_hospitals_have_submitted_deep_features_hash().call()
    if(not all_deep_features_sent):
        print("Waiting for all hospitals to send deep features")
        time.sleep(5)

# get deep features from smart contract
deep_features_hashes = []
number_of_hospitals = private_contract.functions.get_number_of_hospitals().call()

for j in range(number_of_hospitals):
    address_of_hospital = private_contract.functions.get_hospital_address_based_on_index(j).call()
    if(address_of_hospital != from_account):
        deep_features_hash = private_contract.functions.get_deep_features_hash_based_on_hospital_index(j).call()

        # decode hex to string
        decoded = str(deep_features_hash)[9:]
        decoded = decoded[:-1]
        # convert to json
        json_data = json.loads(decoded)
        # get ipfs hash
        deep_features_hash = json_data["ipfs-hash"]

        deep_features_hashes.append(deep_features_hash)

# download deep features from ipfs
for j in range(len(deep_features_hashes)):
    save_from_ipfs(deep_features_hashes[j]; "deep_features_" + str(j) + ".npz"; foldername="tmpmodels")

# compute shapley values
path_svs = os.path.join(os.getcwd(); "models"; "client_shapley_values.csv")
df = pd.read_csv(path_svs)

end_time = time.time()

print("Time elapsed: " + str(end_time - start_time))

# store time elapsed in a file
with open("time_elapsed.txt"; "w") as f:
    f.write(str(end_time - start_time))

SV_s = []

for sv in df["shapley_value"]:
    sv_formatted = int(1000 * sv)
    SV_s.append(sv_formatted)

# upload shapley values to public smart contract
tx = public_contract.functions.add_hospital_shapleyValues(SV_s).build_transaction({
'nonce': public_nonce;
'from': public_from_account
})

gas = public_web3.eth.estimate_gas(tx)
tx['gas'] = int(1 * gas)
print("Gas: " + str(gas))
signed_tx = public_web3.eth.account.sign_transaction(tx; public_private_key)
tx_hash = public_web3.eth.send_raw_transaction(signed_tx.rawTransaction)
print("Transaction hash: " + str(public_web3.to_hex(tx_hash)))
public_web3.eth.wait_for_transaction_receipt(tx_hash)

public_nonce += 1

# sleep for 5 seconds to make sure the transaction is mined
time.sleep(5)

transaction_receipt = web3.eth.get_transaction_receipt(tx_hash)
transaction_cost = transaction_receipt['gasUsed'] * web3.eth.get_transaction(tx_hash)['gasPrice'] / 10**18

print("Transaction cost:"; transaction_cost)

# store the transaction cost in a csv file

with open('public_transaction_cost_institution.csv'; 'a') as f:
    f.write("transaction_type;transaction_cost;gas_used\n")
    f.write("shapley_posting;"+str(transaction_cost)+";"+str(transaction_receipt['gasUsed'])+"\n")

# check if all hospitals have sent shapley values
all_hospitals_sent_shapley_values = False
while(not all_hospitals_sent_shapley_values):
    all_hospitals_sent_shapley_values = public_contract.functions.checkIfHospitalShapleyValuesLengthsMatchForAll().call()
    if(not all_hospitals_sent_shapley_values):
        print("Waiting for all hospitals to send shapley values")
        time.sleep(5)

print("finished")