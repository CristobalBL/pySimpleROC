# roc_curve

How to use

python3 roc_curve.py <clients_file> <fake_clients_file> -n <false_negative_level> -p <false_positive_level>
  
  -p: false positive level to find intersection into ROC Curve
  -n: false negative level to find intersection into ROC Curve

Data Files with scores (clients and fake_clients) must be have the next structure:

< client_id > < score > 
  
The info obtained before run the program is: 
- Quantity of clients, fake clients and scores.
- FNR and threshold when FPR have a fixed value (intercept is showed in ROC curve)
- FPR and threshold when FNR have a fixed value (intercept is showed in ROC curve)
- Values of FPR and FNR in ROC curve when are equals and threshold for that.
- Roc curve area. 
- D Prime value.

