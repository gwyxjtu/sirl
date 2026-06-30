\ Model EventHorizonPlanning
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x50135aa7c0d64721
Maximize
  - 200 Unassigned[0,SCP] - 200 Unassigned[0,PWP] - 200 Unassigned[0,GEP]
   - 200 Unassigned[1,SCP] - 200 Unassigned[1,PWP] - 200 Unassigned[1,GEP]
   - 200 Unassigned[2,SCP] - 200 Unassigned[2,PWP] - 200 Unassigned[2,GEP]
   - 200 Unassigned[3,SCP] - 200 Unassigned[3,PWP] - 200 Unassigned[3,GEP]
   - 200 Unassigned[4,SCP] - 200 Unassigned[4,PWP] - 200 Unassigned[4,GEP]
   - 200 Unassigned[5,SCP] - 200 Unassigned[5,PWP] - 200 Unassigned[5,GEP]
   + 2000 Book[0,SCP] + 1500 Book[0,PWP] + 3000 Book[0,GEP]
   + 2000 Book[1,SCP] + 1500 Book[1,PWP] + 3000 Book[1,GEP]
   + 2000 Book[2,SCP] + 1500 Book[2,PWP] + 3000 Book[2,GEP]
   + 2000 Book[3,SCP] + 1500 Book[3,PWP] + 3000 Book[3,GEP]
   + 2000 Book[4,SCP] + 1500 Book[4,PWP] + 3000 Book[4,GEP]
   + 2000 Book[5,SCP] + 1500 Book[5,PWP] + 3000 Book[5,GEP]
Subject To
 Capacity_Logistics_0: 20 Prepare[0,SCP] + 10 Prepare[0,PWP]
   + 30 Prepare[0,GEP] <= 6400
 Capacity_AudioVisual_0: 10 Prepare[0,SCP] + 20 Prepare[0,GEP] <= 3200
 Capacity_ClientLiaison_0: 15 Prepare[0,PWP] <= 1600
 Capacity_Logistics_1: 20 Prepare[1,SCP] + 10 Prepare[1,PWP]
   + 30 Prepare[1,GEP] <= 4800
 Capacity_AudioVisual_1: 10 Prepare[1,SCP] + 20 Prepare[1,GEP] <= 3200
 Capacity_ClientLiaison_1: 15 Prepare[1,PWP] <= 1600
 Capacity_Logistics_2: 20 Prepare[2,SCP] + 10 Prepare[2,PWP]
   + 30 Prepare[2,GEP] <= 6400
 Capacity_AudioVisual_2: 10 Prepare[2,SCP] + 20 Prepare[2,GEP] <= 3200
 Capacity_ClientLiaison_2: 15 Prepare[2,PWP] <= 1600
 Capacity_Logistics_3: 20 Prepare[3,SCP] + 10 Prepare[3,PWP]
   + 30 Prepare[3,GEP] <= 6400
 Capacity_AudioVisual_3: 10 Prepare[3,SCP] + 20 Prepare[3,GEP] <= 1600
 Capacity_ClientLiaison_3: 15 Prepare[3,PWP] <= 1600
 Capacity_Logistics_4: 20 Prepare[4,SCP] + 10 Prepare[4,PWP]
   + 30 Prepare[4,GEP] <= 6400
 Capacity_AudioVisual_4: 10 Prepare[4,SCP] + 20 Prepare[4,GEP] <= 3200
 Capacity_ClientLiaison_4: 15 Prepare[4,PWP] <= 1600
 Capacity_Logistics_5: 20 Prepare[5,SCP] + 10 Prepare[5,PWP]
   + 30 Prepare[5,GEP] <= 6400
 Capacity_AudioVisual_5: 10 Prepare[5,SCP] + 20 Prepare[5,GEP] <= 3200
 Capacity_ClientLiaison_5: 15 Prepare[5,PWP] <= 1600
 FlowBalance_0_SCP: Prepare[0,SCP] - Unassigned[0,SCP] - Book[0,SCP] = 0
 FlowBalance_1_SCP: Prepare[1,SCP] + Unassigned[0,SCP] - Unassigned[1,SCP]
   - Book[1,SCP] = 0
 FlowBalance_2_SCP: Prepare[2,SCP] + Unassigned[1,SCP] - Unassigned[2,SCP]
   - Book[2,SCP] = 0
 FlowBalance_3_SCP: Prepare[3,SCP] + Unassigned[2,SCP] - Unassigned[3,SCP]
   - Book[3,SCP] = 0
 FlowBalance_4_SCP: Prepare[4,SCP] + Unassigned[3,SCP] - Unassigned[4,SCP]
   - Book[4,SCP] = 0
 FlowBalance_5_SCP: Prepare[5,SCP] + Unassigned[4,SCP] - Unassigned[5,SCP]
   - Book[5,SCP] = 0
 FlowBalance_0_PWP: Prepare[0,PWP] - Unassigned[0,PWP] - Book[0,PWP] = 0
 FlowBalance_1_PWP: Prepare[1,PWP] + Unassigned[0,PWP] - Unassigned[1,PWP]
   - Book[1,PWP] = 0
 FlowBalance_2_PWP: Prepare[2,PWP] + Unassigned[1,PWP] - Unassigned[2,PWP]
   - Book[2,PWP] = 0
 FlowBalance_3_PWP: Prepare[3,PWP] + Unassigned[2,PWP] - Unassigned[3,PWP]
   - Book[3,PWP] = 0
 FlowBalance_4_PWP: Prepare[4,PWP] + Unassigned[3,PWP] - Unassigned[4,PWP]
   - Book[4,PWP] = 0
 FlowBalance_5_PWP: Prepare[5,PWP] + Unassigned[4,PWP] - Unassigned[5,PWP]
   - Book[5,PWP] = 0
 FlowBalance_0_GEP: Prepare[0,GEP] - Unassigned[0,GEP] - Book[0,GEP] = 0
 FlowBalance_1_GEP: Prepare[1,GEP] + Unassigned[0,GEP] - Unassigned[1,GEP]
   - Book[1,GEP] = 0
 FlowBalance_2_GEP: Prepare[2,GEP] + Unassigned[1,GEP] - Unassigned[2,GEP]
   - Book[2,GEP] = 0
 FlowBalance_3_GEP: Prepare[3,GEP] + Unassigned[2,GEP] - Unassigned[3,GEP]
   - Book[3,GEP] = 0
 FlowBalance_4_GEP: Prepare[4,GEP] + Unassigned[3,GEP] - Unassigned[4,GEP]
   - Book[4,GEP] = 0
 FlowBalance_5_GEP: Prepare[5,GEP] + Unassigned[4,GEP] - Unassigned[5,GEP]
   - Book[5,GEP] = 0
 BookingLimit_0_SCP: Book[0,SCP] <= 50
 BookingLimit_0_PWP: Book[0,PWP] <= 80
 BookingLimit_0_GEP: Book[0,GEP] <= 30
 BookingLimit_1_SCP: Book[1,SCP] <= 50
 BookingLimit_1_PWP: Book[1,PWP] <= 80
 BookingLimit_1_GEP: Book[1,GEP] <= 30
 BookingLimit_2_SCP: Book[2,SCP] <= 50
 BookingLimit_2_PWP: Book[2,PWP] <= 80
 BookingLimit_2_GEP: Book[2,GEP] <= 30
 BookingLimit_3_SCP: Book[3,SCP] <= 50
 BookingLimit_3_PWP: Book[3,PWP] <= 80
 BookingLimit_3_GEP: Book[3,GEP] <= 30
 BookingLimit_4_SCP: Book[4,SCP] <= 50
 BookingLimit_4_PWP: Book[4,PWP] <= 80
 BookingLimit_4_GEP: Book[4,GEP] <= 30
 BookingLimit_5_SCP: Book[5,SCP] <= 50
 BookingLimit_5_PWP: Book[5,PWP] <= 80
 BookingLimit_5_GEP: Book[5,GEP] <= 30
 UnassignedLimit_0_SCP: Unassigned[0,SCP] <= 10
 UnassignedLimit_0_PWP: Unassigned[0,PWP] <= 10
 UnassignedLimit_0_GEP: Unassigned[0,GEP] <= 10
 UnassignedLimit_1_SCP: Unassigned[1,SCP] <= 10
 UnassignedLimit_1_PWP: Unassigned[1,PWP] <= 10
 UnassignedLimit_1_GEP: Unassigned[1,GEP] <= 10
 UnassignedLimit_2_SCP: Unassigned[2,SCP] <= 10
 UnassignedLimit_2_PWP: Unassigned[2,PWP] <= 10
 UnassignedLimit_2_GEP: Unassigned[2,GEP] <= 10
 UnassignedLimit_3_SCP: Unassigned[3,SCP] <= 10
 UnassignedLimit_3_PWP: Unassigned[3,PWP] <= 10
 UnassignedLimit_3_GEP: Unassigned[3,GEP] <= 10
 UnassignedLimit_4_SCP: Unassigned[4,SCP] <= 10
 UnassignedLimit_4_PWP: Unassigned[4,PWP] <= 10
 UnassignedLimit_4_GEP: Unassigned[4,GEP] <= 10
 UnassignedLimit_5_SCP: Unassigned[5,SCP] <= 10
 UnassignedLimit_5_PWP: Unassigned[5,PWP] <= 10
 UnassignedLimit_5_GEP: Unassigned[5,GEP] <= 10
 EndOfHorizon_SCP: Unassigned[5,SCP] >= 2
 EndOfHorizon_PWP: Unassigned[5,PWP] >= 3
 EndOfHorizon_GEP: Unassigned[5,GEP] >= 1
Bounds
Generals
 Prepare[0,SCP] Prepare[0,PWP] Prepare[0,GEP] Prepare[1,SCP] Prepare[1,PWP]
 Prepare[1,GEP] Prepare[2,SCP] Prepare[2,PWP] Prepare[2,GEP] Prepare[3,SCP]
 Prepare[3,PWP] Prepare[3,GEP] Prepare[4,SCP] Prepare[4,PWP] Prepare[4,GEP]
 Prepare[5,SCP] Prepare[5,PWP] Prepare[5,GEP] Unassigned[0,SCP]
 Unassigned[0,PWP] Unassigned[0,GEP] Unassigned[1,SCP] Unassigned[1,PWP]
 Unassigned[1,GEP] Unassigned[2,SCP] Unassigned[2,PWP] Unassigned[2,GEP]
 Unassigned[3,SCP] Unassigned[3,PWP] Unassigned[3,GEP] Unassigned[4,SCP]
 Unassigned[4,PWP] Unassigned[4,GEP] Unassigned[5,SCP] Unassigned[5,PWP]
 Unassigned[5,GEP] Book[0,SCP] Book[0,PWP] Book[0,GEP] Book[1,SCP]
 Book[1,PWP] Book[1,GEP] Book[2,SCP] Book[2,PWP] Book[2,GEP] Book[3,SCP]
 Book[3,PWP] Book[3,GEP] Book[4,SCP] Book[4,PWP] Book[4,GEP] Book[5,SCP]
 Book[5,PWP] Book[5,GEP]
End
