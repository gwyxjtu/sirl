\ Model InsuranceSalesStaffScheduling
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x1059c4c4e363f618
Maximize
  Sales[2,2]
Subject To
 MinDemand_Staff2_Period2: WorkTime[2,2] >= 5
 MaxWorkTime_Staff2_Period2: WorkTime[2,2] <= 8
 MaxSales_Staff2_Period2: Sales[2,2] <= 1000
Bounds
 Sales[2,2] <= 1000
 WorkTime[2,2] <= 8
End
