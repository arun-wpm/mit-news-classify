"""
Created Wednesday April 9 2021 11:45 -0400

@author: arunwpm
"""
import csv
def loadcsv(filename):
    with open(filename, newline='', encoding='utf-8') as f: 
        return list(csv.reader(f))