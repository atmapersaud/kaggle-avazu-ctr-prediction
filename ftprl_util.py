from csv import DictReader
def bounded_logsig(x):
    return 1.0 / (1.0 + exp(-max(min(x, 35.), -35.))) 

def dimensions(x): #generates all of the indices of a record
    yield 0
    for d in x:
        yield d

def get_record(datafile, num_weights):
    dfile = open(datafile)
    d_read = DictReader(dfile)
    feature_map = enumerate(d_read)
    for record in feature_map:
        click_id = record['id']
        y = 0  #if test data we'll return 0 by default
        if 'click' in record: # for the training data
            if record['click'] == '1':
                y = 1
        x = []
        for column in record:
            val = record[column]
            hlst = [column, val]
            hstr = '_'.join(hlst)
            hval = hash(hstr)
            idx = abs(hval) % num_weights
            x.append(idx)

    return click_id, x, y
