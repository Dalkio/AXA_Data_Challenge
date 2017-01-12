import numpy as np
from feature_engeneering import features_transformation, init
from prediction import *

# Useless now
"""
# The two ass_assignement that tend to be out of the bucket
outsiders = ['T\xe9l\xe9phonie', 'Tech. Axa']
"""

# define statically the number of lines on the input file
NUMBER_LINES = 82910

_, namesDico = init()

def apply_predict(datetime, ass_assignment, model):
    tok = features_transformation(datetime, ass_assignment, namesDico)
    featuresTransfo = tok.split(';')
    for i in range(len(featuresTransfo)):
        featuresTransfo[i] = float(featuresTransfo[i])
    featuresTransfo = np.asarray(featuresTransfo)

    calls, _ = predict(featuresTransfo, model, cross_val=False)
    result = calls[0][0]
    """ Simple correction if by any way the predicted calls value is negative """
    if result < 0:
        result = 0
    
    return result

if __name__ == '__main__':
    inputfile = 'outputs/submission.txt'
    outputfile = 'outputs/output.txt'
    model = training('data/data_transformed.csv')

    with open(inputfile, 'r') as input, open(outputfile, 'w') as output:
        output.write(input.readline())
        
        compteur = 0.

        for row in input:
            date, ass_assignment, _ = row.split('\t')
            calls = apply_predict(date, ass_assignment, model)
            output.write("{date}\t{ass_assignment}\t{calls}\r".format(date=date, ass_assignment=ass_assignment, calls=calls))
            compteur += 1
            if compteur % 1000 == 0:
                state = compteur / NUMBER_LINES * 100
                print("{0:.2f}".format(state) + '%')