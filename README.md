# qii
QII Code originally from Datta-Sen-Zick Oakland'16  

To try on the adult dataset run:  
python qii.py adult --show-plot

To see additional options:  
Run python qii.py -h

Currently supported datasets:  
adult  : UCI Income dataset  
iwpc   : Warfarin dosage  
nlsy97 : Arrest prediction from the NLSY 97  

Currently supported measures:  
discrim-inf       : Influence on discrimination  
average-local-inf : Average unary QII  
general-inf       : Influence on average classification  
shapley           : Shapley QII (use -i k) for kth individual  
banzhaf           : Banzhaf QII (use -i k) for kth individual  


