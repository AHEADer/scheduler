from datetime import datetime


def log_print(file, msg):
    f = open(file, 'a')
    dateTimeObj = datetime.now()
    f.write(str(dateTimeObj) + ' ' +msg + '\n')
    f.close()
