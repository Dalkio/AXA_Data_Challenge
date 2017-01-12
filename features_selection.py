import pandas as pd

if __name__ == '__main__':
    inputfile = 'data/train_2011_2012_2013.csv'
    outputfile = 'data/data_reduced.csv'

    data = pd.read_csv(inputfile, sep=';', encoding='utf8', usecols=['DATE', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS'])
    data = data[~data['ASS_ASSIGNMENT'].isin(['Evenements', 'Gestion Amex'])] # Those two are not asked to be predicted apparently
    data = data.groupby(['DATE', 'ASS_ASSIGNMENT']).sum().reset_index() # Group and sum values from the same ass_assignment
    data.to_csv(outputfile, sep=';', mode='w', encoding='utf8', index=False)