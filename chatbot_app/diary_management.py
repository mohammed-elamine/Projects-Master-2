from datetime import datetime
import os
import csv
import pandas as pd
from datetime import timedelta
import re

''' return number of copies and pages'''
def return_digits(sentence):
    patterns_paper = 'papers|pages|pg|paper|writing paper|notepaper'
    patterns_copies = 'copies|cp|photocopie|cps|semblance|duplications|semblances'

    number_pages = -1
    number_copies = -1

    find_papers = re.findall('\d+ ' + patterns_paper, sentence)
    if len(find_papers) != 0 : 
        if len(re.findall('\d+', find_papers[0])) != 0:
            number_pages = int(re.findall('\d+', find_papers[0])[0])
    
    find_copies = re.findall('\d+ ' + patterns_copies, sentence)
    if len(find_copies) != 0 :
        if len(re.findall('\d+', find_copies[0])) != 0:
            number_copies = int(re.findall('\d+', find_copies[0])[0])
    
    return (number_pages, number_copies)


''' the duration of printing'''
def duration_calc_sec(pages,copies):
    return pages * copies * 60


''' return the day in the form of  jj_mm_aa '''
def day():
    date = datetime.now()
    day = str(date.day) + '_' + str(date.month) + '_' + str(date.year)
    return day

def date():
    date = datetime.now()
    time = str(date.hour) + ':' + str(date.minute) + ':' + str(date.second)
    return time


''' convert seconds to (h,m,s)'''
def hms(time):
    convert = str(timedelta(seconds = time))
    date = convert.split(':')
    hours = date[0]
    minutes = date[1]
    seconds = date[2]
    return(hours, minutes, seconds)
 

''' if file already exist, it return True, if not it create a new one and return False '''
def check_dir(file_name):
    directory_path = "/".join(file_name.split('/')[:-1])
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)

    return os.path.isfile(file_name)
        

''' calculate the time after the duration of printing'''
def end_day(start_day, duration):
    start=start_day.split(':')
    hours,minutes,seconds=int(start[0]),int(start[1]),int(start[2])
    seconds_start=(hours*60*60)+(minutes*60)+seconds
    seconds_end=seconds_start+duration
    end_day=hms(seconds_end)
    end_day=end_day[0]+':'+end_day[1]+':'+end_day[2]
    return end_day
    

''' calculate the maximum between two dates'''
def max_two_dates(date1,date2):
    date_1= date1.split(':')
    (h1,m1,s1)=(date_1[0],date_1[1],date_1[2])
    date_2=date2.split(':')
    (h2,m2,s2)=(date_2[0],date_2[1],date_2[2])
    if h2>h1 or (h2==h1 and m2>m1) or (h2==h1 and m2== m1 and s2>s1):
        return (h2,m2,s2)
    elif h1>h2 or (h2==h1 and m1>m2) or (h2==h1 and m2==m1 and s1>s2):
        return (h1,m1,s1)
    else : # date1 and date2 are equal
        return (h1,m1,s1)
    

def date1_sup_date2(date1,date2):
    date1 = date1.split(':')
    date2 = date2.split(':')
    seconds1 = (date1[0]*60*60)+ (date1[1]*60)+ date1[2]
    seconds2 = (date2[0]*60*60)+ (date2[1]*60)+ date2[2]
    if seconds1 > seconds2 :
        return True
    else :
        return False
    

''' read the last date in end_day column'''
def extract_end_day(file_name):
    data = pd.read_csv(file_name)
    end_date = data.end[len(data.end)-1]
    return end_date


def generate_file_name():
    jour = day()
    file_name = jour + '.csv'
    return file_name
    

''' the function open a csv file if not already exist, add dates'''
def treat_printing_request(sentence):
    file_name = os.path.join('./diary', generate_file_name())
    boolean = check_dir(file_name)
    
    csv_file = open(file_name, 'a+')
    
    csvWriter = csv.writer(csv_file)
    header = ['start','end']
    #does not exist before
    day_start = date()
    if not boolean:
        csvWriter.writerow(header)
        #day_start=day()
    else:
        day_start = max_two_dates(day_start, extract_end_day(file_name))
        day_start = str(day_start[0]) + ':' + str(day_start[1]) + ':' + str(day_start[2])
    
    (number_pages,number_copies)= return_digits(sentence)
    if ((number_pages == -1) or (number_copies == -1)):
        csv_file.close()
        return (False, 0)
    
    duration = duration_calc_sec(number_pages,number_copies)
    endday=end_day(day_start,duration)
    ## day start and end day  should be before 20h and after 8h
    if (date1_sup_date2(day_start,"20:00:00")==True | date1_sup_date2(endday,"20:00:00")==True | date1_sup_date2("08:00:00",day_start)== True ):
        csv_file.close()
        
        return (False, 1)

    row = [day_start,endday]
    csvWriter.writerow(row)
    csv_file.close()
    return (True, endday)