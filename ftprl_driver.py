import os
from ftprl_classifier import Ftprl_classifier
tr_dir = '/media/owner/storage/ML_Miniproject/traindata/' #directory which holds sharded data files
te_path = '/media/owner/storage/ML_Miniproject/test.csv'
sub_path = '/media/owner/storage/ML_Miniproject/submission.csv'

# hyperparameters
num_epochs = 7
num_params = pow(2,24)
alpha = 0.05
beta = 1
lambda_2 = 1
lambda_1 = 1

classifier = ftprl_classifier(num_params, alpha, beta, lambda_2, lambda_1)

for e in xrange(num_epochs):
    for tr_path in os.listdir(tr_dir):
        for click_id, x, y in get_record(tr_path, num_params):
            yhat = classifier.predict_ctr(x)
            classifier.weight_update(x, y, yhat)

sub_file = open(sub_path, 'w')
sub_file.write('id, click\n')
for click_id, x, y in get_record(te_path, num_params):
    pred = classifier.predict_ctr(x)
    outfile.write('%s, %s\n' % (click_id, str(pred)))



