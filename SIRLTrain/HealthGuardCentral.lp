\ Model HealthGuardCentral
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x11ebf44ede4371e3
Minimize
  15 Procure_A_Q1 + 8 Procure_B_Q1 + 15 Procure_A_Q2 + 8 Procure_B_Q2
   + 15 Procure_A_Q3 + 8 Procure_B_Q3 + 15 Procure_A_Q4 + 8 Procure_B_Q4
   + 0.5 Inventory_A_Q1 + 0.2 Inventory_B_Q1 + 0.5 Inventory_A_Q2
   + 0.2 Inventory_B_Q2 + 0.5 Inventory_A_Q3 + 0.2 Inventory_B_Q3
   + 0.5 Inventory_A_Q4 + 0.2 Inventory_B_Q4 + 100 Shortage_A_Q1
   + 50 Shortage_B_Q1 + 100 Shortage_A_Q2 + 50 Shortage_B_Q2
   + 100 Shortage_A_Q3 + 50 Shortage_B_Q3 + 100 Shortage_A_Q4
   + 50 Shortage_B_Q4
Subject To
 R0: - Procure_A_Q1 + Inventory_A_Q1 + Shortage_A_Q1 = -10000
 R1: - Procure_B_Q1 + Inventory_B_Q1 + Shortage_B_Q1 = -20000
 R2: - Procure_A_Q2 - Inventory_A_Q1 + Inventory_A_Q2 + Shortage_A_Q2
   = -10000
 R3: - Procure_B_Q2 - Inventory_B_Q1 + Inventory_B_Q2 + Shortage_B_Q2
   = -20000
 R4: - Procure_A_Q3 - Inventory_A_Q2 + Inventory_A_Q3 + Shortage_A_Q3
   = -25000
 R5: - Procure_B_Q3 - Inventory_B_Q2 + Inventory_B_Q3 + Shortage_B_Q3
   = -35000
 R6: - Procure_A_Q4 - Inventory_A_Q3 + Inventory_A_Q4 + Shortage_A_Q4
   = -20000
 R7: - Procure_B_Q4 - Inventory_B_Q3 + Inventory_B_Q4 + Shortage_B_Q4
   = -25000
 R8: Inventory_A_Q1 + Inventory_B_Q1 <= 20000
 R9: Inventory_A_Q2 + Inventory_B_Q2 <= 20000
 R10: Inventory_A_Q3 + Inventory_B_Q3 <= 20000
 R11: Inventory_A_Q4 + Inventory_B_Q4 <= 20000
Bounds
 Procure_A_Q1 <= 30000
 Procure_B_Q1 <= 40000
 Procure_A_Q2 <= 30000
 Procure_B_Q2 <= 40000
 Procure_A_Q3 <= 30000
 Procure_B_Q3 <= 40000
 Procure_A_Q4 <= 30000
 Procure_B_Q4 <= 40000
End
