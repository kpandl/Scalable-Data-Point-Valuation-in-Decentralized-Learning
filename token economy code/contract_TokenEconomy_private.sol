// SPDX-License-Identifier: GPL-3.0

pragma solidity 0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract TokenEconomy is Ownable {
   
    struct Hospital {
        bool readyToStart;
        address addr_main; // hospital address
        uint index;   // index of the hospital
        uint[] testResults; // array of test results
        bytes[] modelHashes; // array of model hashes
        bytes deep_feature_hash; // deep feature hash
    }

    address public chairInstitution;

    bytes public initial_model_hash; // array of model hashes

    uint256 public counter;

    //mapping(address => Hospital) public hospitals;

    Hospital[] public hospitals;

    constructor(address[] memory hospitalAddresses, bytes memory _initial_model_hash) {
        chairInstitution = msg.sender;
        initial_model_hash = _initial_model_hash;

        for (uint i = 0; i < hospitalAddresses.length; i++) {
            hospitals.push(Hospital({
                index: i,
                addr_main: hospitalAddresses[i],
                readyToStart: false,
                testResults: new uint[](0),
                modelHashes: new bytes[](0),
                deep_feature_hash: new bytes(0)
            }));
        }
        

    }


  function add_hospital_testResult(uint val) public {
      // check if the hospital is registered
      for (uint i = 0; i < hospitals.length; i++) {
          if (hospitals[i].addr_main == msg.sender) {
              // add test result to hospital
              hospitals[i].testResults.push(val);
              return;
          }
      }
      revert("Hospital not registered.");
  }

  function checkIfHospitalTestResultLengthsMatchForAll() public view returns(bool){
    for (uint i = 0; i < hospitals.length; i++) {
      if (hospitals[i].testResults.length != hospitals[0].testResults.length) {
        return false;
      }
    }
    return true;
  }

  function getLongestHospitalTestResultLength() public view returns(uint){
    uint longestLength = 0;
    for (uint i = 0; i < hospitals.length; i++) {
      if (hospitals[i].testResults.length > longestLength) {
        longestLength = hospitals[i].testResults.length;
      }
    }
    return longestLength;
  }

  function getShortestHospitalTestResultLength() public view returns(uint){
    uint shortestLength = 100000;
    for (uint i = 0; i < hospitals.length; i++) {
      if (hospitals[i].testResults.length < shortestLength) {
        shortestLength = hospitals[i].testResults.length;
      }
    }
    return shortestLength;
  }

  // function that checks if the average test result of all hospitals has not improved for a given number of rounds
  function checkIfHospitalTestResultsShowNoImprovementForTenRounds() public view returns(bool){
    // check if all hospitals have the same number of test results
    if (!checkIfHospitalTestResultLengthsMatchForAll()) {
      return false;
    }
    // get number of rounds from first hospital
    uint numberOfRounds = hospitals[0].testResults.length;
    if(numberOfRounds < 10) {
      return false;
    }
    // empty array of average test results
    uint[] memory averageTestResults = new uint[](numberOfRounds);
    // loop through all rounds
    for (uint i = 0; i < numberOfRounds; i++) {
      // loop through all hospitals
      // compute average test result for a given round
      uint averageTestResult = 0;
      for (uint j = 0; j < hospitals.length; j++) {
        averageTestResult += hospitals[j].testResults[i];
      }
      averageTestResult = averageTestResult / hospitals.length;
      // add average test result to array of average test results
      averageTestResults[i] = averageTestResult;

      // check if average test result has not improved for a given number of rounds
      if (i >= numberOfRounds) {
        if (averageTestResults[i] <= averageTestResults[i - 10]) {
          return false;
        }
      }

    }
    return true;
  }

  function setHospitalReadyToStart() public {
      // check if the hospital is registered
      for (uint i = 0; i < hospitals.length; i++) {
          if (hospitals[i].addr_main == msg.sender) {
              // set hospital ready to start
              hospitals[i].readyToStart = true;
          }
      }
  }

  function checkIfAllHospitalsReadyToStart() public view returns(bool){
    for (uint i = 0; i < hospitals.length; i++) {
      if (!hospitals[i].readyToStart) {
        return false;
      }
    }
    return true;
  }

  function get_number_of_hospitals() public view returns(uint){
    return hospitals.length;
  }

  function get_hospital_ready_to_start_based_on_index(uint index) public view returns(bool){
    return hospitals[index].readyToStart;
  }

  function get_hospital_address_based_on_index(uint index) public view returns(address){
    return hospitals[index].addr_main;
  }

  function store_model_hash(bytes memory modelHash) public {
    // check if the hospital is registered
    for (uint i = 0; i < hospitals.length; i++) {
        if (hospitals[i].addr_main == msg.sender) {
            // add model hash to hospital
            hospitals[i].modelHashes.push(modelHash);
            return;
        }
    }
    revert("Hospital not registered.");
  }

  function get_model_hash_based_on_hospital_index_and_round(uint hospitalIndex, uint round) public view returns(bytes memory){
    return hospitals[hospitalIndex].modelHashes[round];
  }

  function check_if_model_hash_length_is_equal_for_all_hospitals() public view returns(bool){
    uint modelHashLength = hospitals[0].modelHashes.length;
    for (uint i = 0; i < hospitals.length; i++) {
      if (hospitals[i].modelHashes.length != modelHashLength) {
        return false;
      }
    }
    return true;
  }

  function store_deep_features_hash(bytes memory deepFeaturesHash) public {
    // check if the hospital is registered
    for (uint i = 0; i < hospitals.length; i++) {
        if (hospitals[i].addr_main == msg.sender) {
            // add deep features hash to hospital
            hospitals[i].deep_feature_hash = deepFeaturesHash;
            return;
        }
    }
    revert("Hospital not registered.");
  }

  function check_if_all_hospitals_have_submitted_deep_features_hash() public view returns(bool){
    for (uint i = 0; i < hospitals.length; i++) {
      if (hospitals[i].deep_feature_hash.length == 0) {
        return false;
      }
    }
    return true;
  }

  function get_deep_features_hash_based_on_hospital_index(uint hospitalIndex) public view returns(bytes memory){
    return hospitals[hospitalIndex].deep_feature_hash;
  }
    
}
