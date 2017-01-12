import numpy as np
import math
from datetime import datetime
import platform
from workalendar.europe import France
from datetime import *


system = platform.system()
calendar = France()

# define statically the number of lines on the input file
NUMBER_LINES = 1016661

def features_transformation(datetime, ass_assignment, namesDico):
    """ Transform the two given features into two arrays """
    
    dateTransfo = ';'.join([str(x) for x in date_transformation(datetime)])
    nameTransfo = ';'.join([str(x) for x in namesDico[ass_assignment]])
    
    return dateTransfo + ';' + nameTransfo

def date_transformation(date):
    date = datetime.strptime(date[:-4], '%Y-%m-%d %H:%M:%S')
    yesterday = date - timedelta(days=1)
    tomorrow = date + timedelta(days=1)
    days = [0, 0, 0, 0, 0, 0, 0]
    months = [0]*12
    

    days[date.weekday()] = 1

    dayslot = 2 * date.hour
    
    if date.minute < 30:
        dayslot += 1 # add one if in the first half-hour
    else :
        dayslot += 2 # add two if in the second half-hour
    
    months[date.month - 1] = 1
    
    weekend = int(date.weekday() >= 5)
    holiday = int(calendar.is_holiday(date.date()))
    afterholiday = int(calendar.is_holiday(yesterday.date()))
    beforeholiday = int(calendar.is_holiday(tomorrow.date()))
    
    time = int((date - datetime.strptime('2011-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')).total_seconds() / 1800)
    
    return [date.year] + months + [date.day, weekend, holiday, afterholiday, beforeholiday, time, dayslot] + days

def init():
    locations = []
    if system == 'Windows':
        locations = ['Crises', 'CMS', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique',
                     'Gestion Assurances', 'Gestion Clients', 'Gestion DZ', 'Gestion Relation Clienteles',
                     'Gestion Renault', 'Japon', 'Manager', 'M\xc3\xa9canicien', 'M\xc3\xa9dical', 'Nuit',
                     'Prestataires', 'RENAULT', 'RTC', 'Regulation Medicale', 'SAP', 'Services',
                     'Tech. Axa', 'Tech. Inter', 'Tech. Total', 'T\xc3\xa9l\xc3\xa9phonie', 'CAT']
    else: 
        locations = ['Crises', 'CMS', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique','Gestion Assurances', 
                     'Gestion Clients', 'Gestion DZ', 'Gestion Relation Clienteles','Gestion Renault', 'Japon', 'Manager', 
                     'M\xe9canicien', 'M\xe9dical', 'Nuit', 'Prestataires', 'RENAULT', 'RTC', 'Regulation Medicale', 'SAP', 'Services',
                     'Tech. Axa', 'Tech. Inter', 'Tech. Total', 'T\xe9l\xe9phonie', 'CAT']
         
    namesDico = dict(list(zip(locations, [list(x) for x in np.eye(len(locations))[:, 0:]])))
    
    return locations, namesDico

if __name__ == '__main__':
    inputfile = 'data/data_reduced.csv'
    outputfile = 'data/data_transformed.csv'
    
    locations, namesDico = init()
    
    output = open(outputfile, 'w')
    output.write(';'.join((['YEAR', 
             'JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER',
             'DAY', 'WEEKEND', 'HOLIDAY', 'AFTERHOLIDAY', 'BEFOREHOLIDAY', 'TIMESLOT', 'DAYSLOT'] 
             + ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'SUNDAY'] + locations[0:] + ['CALLS'])))
    
    compteur = 0.0
    with open(inputfile, 'r') as data:
        data.readline()
        
        for line in data:
            date, name, calls = line.split(';')
            featuresTransfo = features_transformation(date, name, namesDico)
            output.write(featuresTransfo + ';' + calls)
            compteur += 1
            if compteur % 10000 == 0:
                state = compteur / NUMBER_LINES * 100
                print("{0:.2f}".format(state) + '%')                