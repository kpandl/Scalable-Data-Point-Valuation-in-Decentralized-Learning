import os

file_names = ["private_contract_address.txt", "public_contract_address.txt", "public_transaction_cost.csv", "public_transaction_cost_institution.csv", "public_transaction_cost_orchestrator.csv"]

for file_name in file_names:
    file_path = os.path.join(os.getcwd(), file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        print("Removed " + file_name)