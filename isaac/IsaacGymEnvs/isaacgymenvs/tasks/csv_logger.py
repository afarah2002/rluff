import csv 
import os

class CSVLogger():
    def __init__(self, filename = 'test.csv', fields = ['C1', 'C2'], test_name=""):
        path_name = 'logs/' + test_name + "/"
        os.makedirs(os.path.dirname(path_name), exist_ok=True)
        self.f = open(path_name + filename, 'w')
        self.csvwriter = csv.writer(self.f) 
        # writing the fields 
        self.csvwriter.writerow(fields) 
            

    def write(self, data):
        self.csvwriter.writerows(data)

    def close(self):
        self.f.close()
