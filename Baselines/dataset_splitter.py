# Description: This script splits the original dataset in a train, dev and test set
# Creator: Remi

#!/usr/bin/env python

import os
import csv
import argparse
import re

def read_data():
    ''' Read the original data '''
    sentences = []
    labels = []
    os.chdir('../Data')
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if name[:11] == 'En-Subtask1':
                file = open(os.path.join(root, name))
                text = list(csv.reader(file, delimiter='\t'))
                for row in text[1:]:
                    sentences.append(row[1])
                    labels.append(row[2])
    return sentences, labels

def main():
    X_full, Y_full = read_data()
    print(len(X_full))

    t1 = re.compile(r"I like [\w'-]+ , but not [\w'-]+ .")
    t2 = re.compile(r"I like [\w'-]+ , and [\w'-]+ too .")
    t3 = re.compile(r"I like [\w'-]+ more than [\w'-]+ .")
    t4 = re.compile(r"I do not like [\w'-]+ , I prefer [\w'-]+ .")
    t5 = re.compile(r"I use [\w'-]+ , except [\w'-]+ .")
    t6 = re.compile(r"He likes [\w'-]+ , and [\w'-]+ too .")
    t7 = re.compile(r"He trusts his [\w'-]+ , except [\w'-]+ .")
    t8 = re.compile(r"I like [\w'-]+ , except [\w'-]+ .")
    t9 = re.compile(r"I use [\w'-]+ , and [\w'-]+ too .")
    t10 = re.compile(r"He does not trust [\w'-]+ , he prefers his [\w'-]+ .")
    t11 = re.compile(r"He does not like [\w'-]+ , he prefers [\w'-]+ .")
    t12 = re.compile(r"He trusts his [\w'-]+ more than [\w'-]+ .")
    t13 = re.compile(r"He trusts his [\w'-]+ , but not [\w'-]+ .")
    t14 = re.compile(r"I use [\w'-]+ more than [\w'-]+ .")
    t15 = re.compile(r"I use [\w'-]+ , but not [\w'-]+ .")
    t16 = re.compile(r"I met [\w'-]+ , and [\w'-]+ too .")
    t17 = re.compile(r"He likes [\w'-]+ more than [\w'-]+ .")
    t18 = re.compile(r"He does not trust his [\w'-]+ , he prefers [\w'-]+ .")
    t19 = re.compile(r"I met [\w'-]+ , except [\w'-]+ .")
    t20 = re.compile(r"I met [\w'-]+ , but not [\w'-]+ .")
    t21 = re.compile(r"He trusts his [\w'-]+ , and [\w'-]+ too .")
    t22 = re.compile(r"He likes [\w'-]+ , but not [\w'-]+ .")
    t23 = re.compile(r"He trusts [\w'-]+ , except his [\w'-]+ .")   
    t24 = re.compile(r"He likes [\w'-]+ , except [\w'-]+ .")
    t25 = re.compile(r"He trusts [\w'-]+ more than his [\w'-]+ .")
    t26 = re.compile(r"He trusts [\w'-]+ , but not his [\w'-]+ .")
    t27 = re.compile(r"He trusts his [\w'-]+ , and his [\w'-]+ too .")
    t28 = re.compile(r"He trusts [\w'-]+ , and his [\w'-]+ too .")
    
    t29 = re.compile(r"I like [\w'-]+ , an interesting type of [\w'-]+ .")
    t30 = re.compile(r"He trusts his [\w'-]+ , an interesting type of [\w'-]+ .")
    t31 = re.compile(r"He likes [\w'-]+ , an interesting type of [\w'-]+ .")
    t32 = re.compile(r"I met [\w'-]+ , an interesting type of [\w'-]+ .")
    t33 = re.compile(r"He trusts [\w'-]+ , an interesting type of [\w'-]+ .")
    t34 = re.compile(r"I use [\w'-]+ , an interesting type of [\w'-]+ .")

    t35 = re.compile(r"I like [\w'-]+ , and more specifically [\w'-]+ .")
    t36 = re.compile(r"I use [\w'-]+ , and more specifically [\w'-]+ .")
    t37 = re.compile(r"I met [\w'-]+ , and more specifically [\w'-]+ .")
    t38 = re.compile(r"He trusts [\w'-]+ , and more specifically his [\w'-]+ .")
    t39 = re.compile(r"He trusts his [\w'-]+ , and more specifically [\w'-]+ .")
    t40 = re.compile(r"He likes [\w'-]+ , and more specifically [\w'-]+ .")

    y = 0
    for s in X_full:
        if t29.match(s) or t30.match(s) or t31.match(s) or t32.match(s) or t33.match(s) or t34.match(s):
            f = open("dev.txt", "a+")
            f.write("{} {}\n".format(s, Y_full[y]))
        elif t35.match(s) or t36.match(s) or t37.match(s) or t38.match(s) or t39.match(s) or t40.match(s):
            f = open("test.txt", "a+")
            f.write("{} {}\n".format(s, Y_full[y]))
        else:
            f = open("train.txt", "a+")
            f.write("{} {}\n".format(s, Y_full[y]))
        y+=1
    print(y)

if __name__ == "__main__":
    main()
