\ Model ChainLearn Curriculum Optimization
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x9fe2e530fc5d6cc9
Maximize
  40 select_module[M1] + 55 select_module[M2] + 35 select_module[M3]
   + 60 select_module[M4] + 48 select_module[M5] + 42 select_module[M6]
   + 50 select_module[M7] + 65 select_module[M8]
Subject To
 Learning_Load_Limit: 30 select_module[M1] + 45 select_module[M2]
   + 25 select_module[M3] + 50 select_module[M4] + 40 select_module[M5]
   + 35 select_module[M6] + 40 select_module[M7] + 55 select_module[M8]
   <= 150
 Solidity_Expert_Limit: 15 select_module[M1] + 10 select_module[M3] <= 80
 DeFi_Expert_Limit: 20 select_module[M2] <= 60
 Security_Expert_Limit: 25 select_module[M4] + 30 select_module[M8] <= 100
 Scaling_Expert_Limit: 18 select_module[M5] <= 50
 Full-stack_Web3_Expert_Limit: 15 select_module[M6] <= 70
 Hyperledger_Expert_Limit: 20 select_module[M7] <= 50
 Ethereum_Dev_Lab_Limit: 20 select_module[M1] + 30 select_module[M2]
   + 15 select_module[M3] + 25 select_module[M5] <= 150
 Shared_Security_Lab_Limit: 35 select_module[M4] + 40 select_module[M8]
   <= 120
 Shared_Web3_Lab_Limit: 20 select_module[M6] <= 80
 Hyperledger_Dev_Lab_Limit: 25 select_module[M7] <= 100
 Security_Expert_Required: select_module[M4] + select_module[M8] >= 1
 Ethereum_Lab_Required: select_module[M1] + select_module[M2]
   + select_module[M3] + select_module[M5] >= 1
 Max_Modules: select_module[M1] + select_module[M2] + select_module[M3]
   + select_module[M4] + select_module[M5] + select_module[M6]
   + select_module[M7] + select_module[M8] <= 4
Bounds
Binaries
 select_module[M1] select_module[M2] select_module[M3] select_module[M4]
 select_module[M5] select_module[M6] select_module[M7] select_module[M8]
End
