\ Model Supply_Chain_Network_Design
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xb3a55355edf9f21e
Minimize
  10 x[1,1] + 20 x[1,2] + 15 x[2,1] + 25 x[2,2] + 500 y[1] + 800 y[2]
   + 5 I[1] + 8 I[2]
Subject To
 Capacity_DC_1: x[1,1] + x[1,2] - 100 y[1] <= 0
 Capacity_DC_2: x[2,1] + x[2,2] - 200 y[2] <= 0
 Demand_Store_1: x[1,1] + x[2,1] >= 150
 Demand_Store_2: x[1,2] + x[2,2] >= 100
 Min_Safety_Stock_DC_1: I[1] >= 20
 Min_Safety_Stock_DC_2: I[2] >= 30
Bounds
Binaries
 y[1] y[2]
End
