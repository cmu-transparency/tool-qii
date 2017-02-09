# Set working directory
# setwd()

new_data <- read.table('default.dat', sep=' ')
names(new_data) <- c('R0000100','R0357900','R0536300','R0536401','R0536402','R1200300','R1205300','R1235800','R1482600','R2189100','R2200500','R2558800','R2563600','R3508200','R3511100','R3520100','R3880300','R3885200','R4909200','R4918000','R5459400','R5464400','R6536400','R6545000','R7222400','R7228100','S0924000','S0932900','S1535500','S1542000','S2005400','S2011800','S2990300','S2995900','S3805700','S4682600','S4685500','S4691000','S5405600','S6318100','S6321000','S6326800','S7506100','S8334600','S8339100','T0009400','T0741400','T0745900','T2012100','T2784900','T2789700','T3602100','T4496600','T4501300','T5202300','T6145500','T6150200','T6652100','T7640000','T7644700','T8123700','T9044600','Z9050500','Z9050600','Z9050700')

# Handle missing values
  new_data[new_data == -1] = NA  # Refused 
  new_data[new_data == -2] = NA  # Dont know 
  new_data[new_data == -3] = NA  # Invalid missing 
  new_data[new_data == -4] = NA  # Valid missing 
  new_data[new_data == -5] = NA  # Non-interview 

# If there are values not categorized they will be represented as NA
vallabels = function(data) {
  data$R0000100 <- cut(data$R0000100, c(0.0,1.0,1000.0,2000.0,3000.0,4000.0,5000.0,6000.0,7000.0,8000.0,9000.0,9999.0), labels=c("0","1 TO 999","1000 TO 1999","2000 TO 2999","3000 TO 3999","4000 TO 4999","5000 TO 5999","6000 TO 6999","7000 TO 7999","8000 TO 8999","9000 TO 9999"), right=FALSE)
  data$R0357900 <- factor(data$R0357900, levels=c(1.0,0.0), labels=c("Yes","No"))
  data$R0536300 <- factor(data$R0536300, levels=c(1.0,2.0,0.0), labels=c("Male","Female","No Information"))
  data$R0536401 <- factor(data$R0536401, levels=c(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0), labels=c("1: January","2: February","3: March","4: April","5: May","6: June","7: July","8: August","9: September","10: October","11: November","12: December"))
  data$R1200300 <- factor(data$R1200300, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$R1205300 <- factor(data$R1205300, levels=c(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0), labels=c("Both biological parents","Two parents, biological mother","Two parents, biological father","Biological mother only","Biological father only","Adoptive parent(s)","Foster parent(s)","No parents, grandparents","No parents, other relatives","Anything else"))
  data$R1235800 <- factor(data$R1235800, levels=c(1.0,0.0), labels=c("Cross-sectional","Oversample"))
  data$R1482600 <- factor(data$R1482600, levels=c(1.0,2.0,3.0,4.0), labels=c("Black","Hispanic","Mixed Race (Non-Hispanic)","Non-Black / Non-Hispanic"))
  data$R2189100 <- factor(data$R2189100, levels=c(1.0,0.0), labels=c("Yes","No"))
  data$R2200500 <- cut(data$R2200500, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,99.0), labels=c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15 TO 99: 15+"), right=FALSE)
  data$R2558800 <- factor(data$R2558800, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$R2563600 <- factor(data$R2563600, levels=c(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0), labels=c("Both biological parents","Two parents, biological mother","Two parents, biological father","Biological mother only","Biological father only","Adoptive parent(s)","Foster parent(s)","No parents, grandparents","No parents, other relatives","Anything else"))
  data$R3508200 <- factor(data$R3508200, levels=c(1.0,0.0), labels=c("Yes","No"))
  data$R3511100 <- factor(data$R3511100, levels=c(1.0,0.0), labels=c("Yes","No"))
  data$R3520100 <- cut(data$R3520100, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,99.0), labels=c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15 TO 99: 15+"), right=FALSE)
  data$R3880300 <- factor(data$R3880300, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$R3885200 <- factor(data$R3885200, levels=c(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0), labels=c("Both biological parents","Two parents, biological mother","Two parents, biological father","Biological mother only","Biological father only","Adoptive parent(s)","Foster parent(s)","No parents, grandparents","No parents, other relatives","Anything else"))
  data$R4909200 <- factor(data$R4909200, levels=c(1.0,0.0), labels=c("Yes","No"))
  data$R4918000 <- cut(data$R4918000, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15 TO 999: 15+"), right=FALSE)
  data$R5459400 <- factor(data$R5459400, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$R5464400 <- factor(data$R5464400, levels=c(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0), labels=c("Both biological parents","Two parents, biological mother","Two parents, biological father","Biological mother only","Biological father only","Adoptive parent(s)","Foster parent(s)","No parents, grandparents","No parents, other relatives","Anything else"))
  data$R6536400 <- factor(data$R6536400, levels=c(1.0,0.0), labels=c("YES","NO"))
  data$R6545000 <- cut(data$R6545000, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15 TO 999: 15+"), right=FALSE)
  data$R7222400 <- factor(data$R7222400, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$R7228100 <- factor(data$R7228100, levels=c(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0), labels=c("Both biological parents","Two parents, biological mother","Two parents, biological father","Biological mother only","Biological father only","Adoptive parent(s)","Foster parent(s)","No parents, grandparents","No parents, other relatives","Anything else"))
  data$S0924000 <- factor(data$S0924000, levels=c(1.0,0.0), labels=c("YES","NO"))
  data$S0932900 <- cut(data$S0932900, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15 TO 999: 15+"), right=FALSE)
  data$S1535500 <- factor(data$S1535500, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$S1542000 <- factor(data$S1542000, levels=c(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0), labels=c("Both biological parents","Two parents, biological mother","Two parents, biological father","Biological mother only","Biological father only","Adoptive parent(s)","Foster parent(s)","No parents, grandparents","No parents, other relatives","Anything else"))
  data$S2005400 <- factor(data$S2005400, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$S2011800 <- factor(data$S2011800, levels=c(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0), labels=c("Both biological parents","Two parents, biological mother","Two parents, biological father","Biological mother only","Biological father only","Adoptive parent(s)","Foster parent(s)","No parents, grandparents","No parents, other relatives","Anything else"))
  data$S2990300 <- factor(data$S2990300, levels=c(1.0,0.0), labels=c("YES","NO"))
  data$S2995900 <- cut(data$S2995900, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15 TO 999: 15+"), right=FALSE)
  data$S3805700 <- factor(data$S3805700, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$S4682600 <- factor(data$S4682600, levels=c(1.0,0.0), labels=c("YES","NO"))
  data$S4685500 <- factor(data$S4685500, levels=c(1.0,0.0), labels=c("YES","NO"))
  data$S4691000 <- cut(data$S4691000, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15 TO 999: 15+"), right=FALSE)
  data$S5405600 <- factor(data$S5405600, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$S6318100 <- factor(data$S6318100, levels=c(1.0,0.0), labels=c("YES","NO"))
  data$S6321000 <- factor(data$S6321000, levels=c(1.0,0.0), labels=c("YES","NO"))
  data$S6326800 <- cut(data$S6326800, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15 TO 999: 15+"), right=FALSE)
  data$S7506100 <- factor(data$S7506100, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$S8334600 <- factor(data$S8334600, levels=c(1.0,0.0), labels=c("YES","NO"))
  data$S8339100 <- cut(data$S8339100, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15 TO 999: 15+"), right=FALSE)
  data$T0009400 <- factor(data$T0009400, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$T0741400 <- factor(data$T0741400, levels=c(1.0,0.0), labels=c("YES","NO"))
  data$T0745900 <- cut(data$T0745900, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15 TO 999: 15+"), right=FALSE)
  data$T2012100 <- factor(data$T2012100, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$T2784900 <- factor(data$T2784900, levels=c(1.0,0.0), labels=c("YES","NO"))
  data$T2789700 <- cut(data$T2789700, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15 TO 999: 15+"), right=FALSE)
  data$T3602100 <- factor(data$T3602100, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$T4496600 <- factor(data$T4496600, levels=c(1.0,0.0), labels=c("YES","NO"))
  data$T4501300 <- cut(data$T4501300, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15 TO 999: 15+"), right=FALSE)
  data$T5202300 <- factor(data$T5202300, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$T6145500 <- factor(data$T6145500, levels=c(1.0,0.0), labels=c("YES","NO"))
  data$T6150200 <- cut(data$T6150200, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15 TO 999: 15+"), right=FALSE)
  data$T6652100 <- factor(data$T6652100, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$T7640000 <- factor(data$T7640000, levels=c(1.0,0.0), labels=c("YES","NO"))
  data$T7644700 <- cut(data$T7644700, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15 TO 999: 15+"), right=FALSE)
  data$T8123700 <- factor(data$T8123700, levels=c(1.0,2.0,3.0,4.0), labels=c("Northeast (CT, ME, MA, NH, NJ, NY, PA, RI, VT)","North Central (IL, IN, IA, KS, MI, MN, MO, NE, OH, ND, SD, WI)","South (AL, AR, DE, DC, FL, GA, KY, LA, MD, MS, NC, OK, SC, TN , TX, VA, WV)","West (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)"))
  data$T9044600 <- cut(data$T9044600, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15 TO 999: 15+"), right=FALSE)
  data$Z9050500 <- cut(data$Z9050500, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10 TO 999: 10+"), right=FALSE)
  data$Z9050600 <- cut(data$Z9050600, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10 TO 999: 10+"), right=FALSE)
  data$Z9050700 <- cut(data$Z9050700, c(0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,999.0), labels=c("0","1","2","3","4","5","6","7","8","9","10 TO 999: 10+"), right=FALSE)
  return(data)
}

varlabels <- c(    "PUBID - YTH ID CODE 1997",
    "R EVER SMOKE? 1997",
    "KEY!SEX (SYMBOL) 1997",
    "KEY!BDATE M/Y (SYMBOL) 1997",
    "KEY!BDATE M/Y (SYMBOL) 1997",
    "CV_CENSUS_REGION 1997",
    "CV_YTH_REL_HH_CURRENT 1997",
    "CV_SAMPLE_TYPE 1997",
    "KEY!RACE_ETHNICITY (SYMBOL) 1997",
    "R EVER SMOKE? 1998",
    "TTL # TIMES ARRESTED SDLI 1998",
    "CV_CENSUS_REGION 1998",
    "CV_YTH_REL_HH_CURRENT 1998",
    "R EVER SMOKE? 1999",
    "R USE COC/DRUGS SDLI? 1999",
    "TTL # TIMES ARRESTED SDLI 1999",
    "CV_CENSUS_REGION 1999",
    "CV_YTH_REL_HH_CURRENT 1999",
    "R USE COC/DRUGS SDLI? 2000",
    "TTL # TIMES ARRESTED SDLI 2000",
    "CV_CENSUS_REGION 2000",
    "CV_YTH_REL_HH_CURRENT 2000",
    "R USE COC/DRUGS SDLI? 2001",
    "TTL # TIMES ARRESTED SDLI 2001",
    "CV_CENSUS_REGION 2001",
    "CV_YTH_REL_HH_CURRENT 2001",
    "R USE COC/DRUGS SDLI? 2002",
    "TTL # TIMES ARRESTED SDLI 2002",
    "CV_CENSUS_REGION 2002",
    "CV_YTH_REL_HH_CURRENT 2002",
    "CV_CENSUS_REGION 2003",
    "CV_YTH_REL_HH_CURRENT 2003",
    "R USE COC/DRUGS SDLI? 2003",
    "TTL # TIMES ARRESTED SDLI 2003",
    "CV_CENSUS_REGION 2004",
    "R EVER SMOKE? 2004",
    "R USE COC/DRUGS SDLI? 2004",
    "TTL # TIMES ARRESTED SDLI 2004",
    "CV_CENSUS_REGION 2005",
    "R EVER SMOKE? 2005",
    "R USE COC/DRUGS SDLI? 2005",
    "TTL # TIMES ARRESTED SDLI 2005",
    "CV_CENSUS_REGION 2006",
    "R USE COC/DRUGS SDLI? 2006",
    "TTL # TIMES ARRESTED SDLI 2006",
    "CV_CENSUS_REGION 2007",
    "R USE COC/DRUGS SDLI? 2007",
    "TTL # TIMES ARRESTED SDLI 2007",
    "CV_CENSUS_REGION 2008",
    "R USE COC/DRUGS SDLI? 2008",
    "TTL # TIMES ARRESTED SDLI 2008",
    "CV_CENSUS_REGION 2009",
    "R USE COC/DRUGS SDLI? 2009",
    "TTL # TIMES ARRESTED SDLI 2009",
    "CV_CENSUS_REGION 2010",
    "R USE COC/DRUGS SDLI? 2010",
    "TTL # TIMES ARRESTED SDLI 2010",
    "CV_CENSUS_REGION 2011",
    "R USE COC/DRUGS SDLI? 2011",
    "TTL # TIMES ARRESTED SDLI 2011",
    "CV_CENSUS_REGION 2013",
    "TTL # TIMES ARRESTED SDLI 2013",
    "CVC_TTL_JOB_TEEN",
    "CVC_TTL_JOB_ADULT_ET",
    "CVC_TTL_JOB_ADULT_ALL"
)

# Use qnames rather than rnums
qnames = function(data) {
  names(data) <- c("PUBID_1997","YSAQ-359_1997","KEY_SEX_1997","KEY_BDATE_M_1997","KEY_BDATE_Y_1997","CV_CENSUS_REGION_1997","CV_YTH_REL_HH_CURRENT_1997","CV_SAMPLE_TYPE_1997","KEY_RACE_ETHNICITY_1997","YSAQ-359_1998","YSAQ-443_1998","CV_CENSUS_REGION_1998","CV_YTH_REL_HH_CURRENT_1998","YSAQ-359_1999","YSAQ-372CC_1999","YSAQ-443_1999","CV_CENSUS_REGION_1999","CV_YTH_REL_HH_CURRENT_1999","YSAQ-372CC_2000","YSAQ-443_2000","CV_CENSUS_REGION_2000","CV_YTH_REL_HH_CURRENT_2000","YSAQ-372CC_2001","YSAQ-443_2001","CV_CENSUS_REGION_2001","CV_YTH_REL_HH_CURRENT_2001","YSAQ-372CC_2002","YSAQ-443_2002","CV_CENSUS_REGION_2002","CV_YTH_REL_HH_CURRENT_2002","CV_CENSUS_REGION_2003","CV_YTH_REL_HH_CURRENT_2003","YSAQ-372CC_2003","YSAQ-443_2003","CV_CENSUS_REGION_2004","YSAQ-359_2004","YSAQ-372CC_2004","YSAQ-443_2004","CV_CENSUS_REGION_2005","YSAQ-359_2005","YSAQ-372CC_2005","YSAQ-443_2005","CV_CENSUS_REGION_2006","YSAQ-372CC_2006","YSAQ-443_2006","CV_CENSUS_REGION_2007","YSAQ-372CC_2007","YSAQ-443_2007","CV_CENSUS_REGION_2008","YSAQ-372CC_2008","YSAQ-443_2008","CV_CENSUS_REGION_2009","YSAQ-372CC_2009","YSAQ-443_2009","CV_CENSUS_REGION_2010","YSAQ-372CC_2010","YSAQ-443_2010","CV_CENSUS_REGION_2011","YSAQ-372CC_2011","YSAQ-443_2011","CV_CENSUS_REGION_2013","YSAQ-443_2013","CVC_TTL_JOB_TEEN_XRND","CVC_TTL_JOB_ADULT_ET_XRND","CVC_TTL_JOB_ADULT_ALL_XRND")
  return(data)
}

********************************************************************************************************

# Remove the '#' before the following line to create a data file called "categories" with value labels. 
#categories <- vallabels(new_data)

# Remove the '#' before the following lines to rename variables using Qnames instead of Reference Numbers
#new_data <- qnames(new_data)
#categories <- qnames(categories)

# Produce summaries for the raw (uncategorized) data file
summary(new_data)

# Remove the '#' before the following lines to produce summaries for the "categories" data file.
#categories <- vallabels(new_data)
#summary(categories)

************************************************************************************************************
