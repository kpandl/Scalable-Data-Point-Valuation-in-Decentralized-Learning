  // SPDX-License-Identifier: GPL-3.0

  pragma solidity 0.8.0;

  import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
  import "@openzeppelin/contracts/access/Ownable.sol";

  contract TokenEconomy is Ownable {
    
      struct Hospital {
          bool readyToStart;
          address addr_main; // hospital address
          uint index;   // index of the hospital
          uint[] shapleyValues; // array of reported shapley values
      }

      address public chairInstitution;
      address erc20_token;

      Hospital[] public hospitals;
      mapping(address => uint256) public userDeposits;

      constructor(address[] memory hospitalAddresses, address _erc20_token) {
          chairInstitution = msg.sender;
          erc20_token = _erc20_token;

          for (uint i = 0; i < hospitalAddresses.length; i++) {
              hospitals.push(Hospital({
                  index: i,
                  addr_main: hospitalAddresses[i],
                  readyToStart: false,
                  shapleyValues: new uint[](0)
              }));
          }
      }


      function deposit_transfer(uint _amount) public payable {
      // Set the minimum amount to 1 token (in this case I'm using LINK token)
      uint _minAmount = 1;//*(10**18);
      // Here we validate if sended USDT for example is higher than 50, and if so we increment the counter_deposits
      require(_amount >= _minAmount, "Amount less than minimum amount");
      // I call the function of IERC20 contract to transfer the token from the user (that he's interacting with the contract) to
      // the smart contract  
      
      IERC20(erc20_token).transferFrom(msg.sender, address(this), _amount);

      userDeposits[msg.sender] += _amount;
    }

    function add_hospital_shapleyValues(uint[] memory shapleyValues) public {
        // check if the hospital is registered
        for (uint i = 0; i < hospitals.length; i++) {
            if (hospitals[i].addr_main == msg.sender) {
                // add shapley values to hospital
                hospitals[i].shapleyValues = shapleyValues;
                return;
            }
        }
        revert("Hospital not registered.");
    }

    function checkIfHospitalShapleyValuesLengthsMatchForAll() public view returns(bool){
      for (uint i = 0; i < hospitals.length; i++) {
        if (hospitals[i].shapleyValues.length != hospitals[0].shapleyValues.length) {
          return false;
        }
      }
      return true;
    }

    // function to get the balance of the smart contract
    function getBalance() public view returns(uint){
      return IERC20(erc20_token).balanceOf(address(this));
    }

    // function to pay the balance to the hospitals
    function payBalanceToAllHospitalsImmediately() public {
      uint[] memory contribution_sums = new uint[](hospitals.length);

      for (uint i = 0; i < hospitals.length; i++) {
        // the hospital that has posted
        for (uint j = 0; j < hospitals[i].shapleyValues.length; j++) {
          // the hospital of the shapley value
          contribution_sums[j] = contribution_sums[j] + hospitals[i].shapleyValues[j];
        }
      }

      uint max_contribution_sum = 500 * hospitals.length;
      uint balance_initial = getBalance();

      for (uint i = 0; i < hospitals.length; i++) {
        IERC20(erc20_token).transfer(hospitals[i].addr_main, balance_initial * contribution_sums[i] / max_contribution_sum);
    }

      uint remaining_balance = getBalance();
      // transfer the remaining balance to the owner
      IERC20(erc20_token).transfer(msg.sender, remaining_balance);
    
  }}