# roc_curve

How to use

python3 roc_curve.py <clients_file> <fake_clients_file> -n <false_negative_level> -p <false_positive_lebel>
  
  -p: false positive level to find intersection into ROC Curve
  -n: false negative level to find intersection into ROC Curve

Data Files with scores (clients and fake_clients) must be have the next structure:

<client_id> <score> 

