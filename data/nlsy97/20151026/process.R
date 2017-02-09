library(NLSdata)
library(data.table)

## Create NLS Data Structures
nls.obj <- CreateNLSdata("default.cdb", "default.csv")
base = nls.obj$data[order(nls.obj$data$PUBID.1997), c("PUBID.1997", "KEY!SEX.1997", "KEY!BDATE_Y.1997", "CV_CENSUS_REGION.1997", "KEY!RACE_ETHNICITY.1997")]

## Create total number of arrested
arrested <- CreateTimeSeriesDf(nls.obj, "YSAQ_443")
arrested[is.na(arrested)] <- 0
arrested_sum <- aggregate(arrested$YSAQ_443, by=list(PUBID.1997=arrested$PUBID.1997), FUN=sum)

dt1 = data.table(base, key = "PUBID.1997")
dt2 = data.table(arrested_sum, key = "PUBID.1997")
joined_dt12 <- dt1[dt2]
colnames(joined_dt12)[6]
colnames(joined_dt12)[6] <- "TOTAL_ARRESTS"

#parents <- CreateTimeSeriesDf(nls.obj, "CV_YTH_REL_HH_CURRENT")

## Create a cumulative smoking variable
smoking <- CreateTimeSeriesDf(nls.obj, "YSAQ_359")
smoking$YSAQ_359[smoking$YSAQ_359 == "No"] = "NO"
smoking$YSAQ_359[smoking$YSAQ_359 == "Yes   (Go To R03580.00)"] = "YES"
smoking$YSAQ_359[smoking$YSAQ_359 == "Yes   (Go To R21893.00)"] = "YES"
smoking$YSAQ_359[smoking$YSAQ_359 == "Yes   (Go To R35084.00)"] = "YES"
smoking$YSAQ_359[smoking$YSAQ_359 == "YES   (Go To S46828.00)"] = "YES"
smoking$YSAQ_359[smoking$YSAQ_359 == NA] = "NO"
smoking$YSAQ_359[is.na(smoking$YSAQ_359)] = "NO"
smoking$YSAQ_359 <- as.factor(smoking$YSAQ_359)
smoking$YSAQ_359 <- as.numeric(smoking$YSAQ_359) - 1
smoking_binary <- aggregate(smoking$YSAQ_359, by=list(PUBID.1997=smoking$PUBID.1997), FUN=max)

## Create a cumulative drugs variable
drugs <- CreateTimeSeriesDf(nls.obj, "YSAQ_372CC")
unique(drugs$YSAQ_372CC)
drugs$YSAQ_372CC <- strtrim(drugs$YSAQ_372CC, 3)
drugs$YSAQ_372CC[is.na(drugs$YSAQ_372CC)] <- "NO"
drugs$YSAQ_372CC[drugs$YSAQ_372CC == "No"] <- "NO"
drugs$YSAQ_372CC[drugs$YSAQ_372CC == "Yes"] <- "YES"
drugs$YSAQ_372CC <- as.numeric(as.factor(drugs$YSAQ_372CC)) - 1
drugs_binary <- aggregate(drugs$YSAQ_372CC, by=list(PUBID.1997=drugs$PUBID.1997), FUN=max)

## Merge tables
dt_drugs = data.table(drugs_binary, key = "PUBID.1997")
dt_smoking = data.table(smoking_binary, key = "PUBID.1997")
colnames(dt_drugs)[2] <- "HAS_DRUGS"
colnames(dt_smoking)[2] <- "HAS_SMOKED"
dt_combined1 <- dt_drugs[dt_smoking]
dt_combined2 <- joined_dt12[dt_combined1]

## Write to CSV
write.table(dt_combined2, file="processed_output1.csv", row.names = FALSE, col.names = FALSE, sep = ", ")
