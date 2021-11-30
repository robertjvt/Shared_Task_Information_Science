#!/usr/bin/env python

import os
import csv
import argparse
import re


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", default='reviews.txt', type=str,
                        help="Input file to learn from (default reviews.txt)")
    args = parser.parse_args()
    return args


def read_data():
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
    args = create_arg_parser()
    X_full, Y_full = read_data()

    t1 = re.compile(r"I like [\w'-]+ , but not [\w'-]+ .")
    t2 = re.compile(r"I like [\w'-]+ , and [\w'-]+ too .")
    t3 = re.compile(r"I like [\w'-]+ more than [\w'-]+ .")
    t4 = re.compile(r"I like [\w'-]+ , an interesting type of [\w'-]+ .")
    t5 = re.compile(r"I do not like [\w'-]+ , I prefer [\w'-]+ .")
    t6 = re.compile(r"I like [\w'-]+ , and more specifically [\w'-]+ .")
    t7 = re.compile(r"I use [\w'-]+ , except [\w'-]+ .")
    t8 = re.compile(r"He likes [\w'-]+ , and [\w'-]+ too .")
    t9 = re.compile(r"I use [\w'-]+ , and more specifically [\w'-]+ .")
    t10 = re.compile(r"I met [\w'-]+ , and more specifically [\w'-]+ .")
    t11 = re.compile(r"He trusts his [\w'-]+ , an interesting type of [\w'-]+ .")
    t12 = re.compile(r"He trusts his [\w'-]+ , except [\w'-]+ .")
    t13 = re.compile(r"I like [\w'-]+ , except [\w'-]+ .")
    t14 = re.compile(r"I use [\w'-]+ , and [\w'-]+ too .")
    t15 = re.compile(r"He trusts [\w'-]+ , and more specifically his [\w'-]+ .")
    t16 = re.compile(r"He likes [\w'-]+ , an interesting type of [\w'-]+ .")
    t17 = re.compile(r"He does not trust [\w'-]+ , he prefers his [\w'-]+ .")
    t18 = re.compile(r"I met [\w'-]+ , an interesting type of [\w'-]+ .")
    t19 = re.compile(r"He does not like [\w'-]+ , he prefers [\w'-]+ .")
    t20 = re.compile(r"He trusts his [\w'-]+ more than [\w'-]+ .")
    t21 = re.compile(r"He trusts [\w'-]+ , an interesting type of [\w'-]+ .")
    t22 = re.compile(r"He trusts his [\w'-]+ , but not [\w'-]+ .")
    t23 = re.compile(r"I use [\w'-]+ more than [\w'-]+ .")
    t24 = re.compile(r"I use [\w'-]+ , but not [\w'-]+ .")
    t25 = re.compile(r"I met [\w'-]+ , and [\w'-]+ too .")
    t26 = re.compile(r"I use [\w'-]+ , an interesting type of [\w'-]+ .")
    t27 = re.compile(r"He likes [\w'-]+ more than [\w'-]+ .")
    t28 = re.compile(r"He does not trust his [\w'-]+ , he prefers [\w'-]+ .")
    t29 = re.compile(r"I met [\w'-]+ , except [\w'-]+ .")
    t30 = re.compile(r"I met [\w'-]+ , but not [\w'-]+ .")
    t31 = re.compile(r"He trusts his [\w'-]+ , and [\w'-]+ too .")
    t32 = re.compile(r"He likes [\w'-]+ , but not [\w'-]+ .")
    t33 = re.compile(r"He trusts his [\w'-]+ , and more specifically [\w'-]+ .")
    t34 = re.compile(r"He trusts [\w'-]+ , except his [\w'-]+ .")
    t35 = re.compile(r"He likes [\w'-]+ , and more specifically [\w'-]+ .")
    t36 = re.compile(r"He likes [\w'-]+ , except [\w'-]+ .")
    t37 = re.compile(r"He trusts [\w'-]+ more than his [\w'-]+ .")
    t38 = re.compile(r"He trusts [\w'-]+ , but not his [\w'-]+ .")
    t39 = re.compile(r"He trusts his [\w'-]+ , and his [\w'-]+ too .")
    t40 = re.compile(r"He trusts [\w'-]+ , and his [\w'-]+ too .")

    t1_counter = t2_counter = t3_counter = t4_counter = t5_counter = t6_counter = t7_counter = t8_counter = t9_counter = t10_counter = t11_counter = t12_counter = 0
    t13_counter = t14_counter = t15_counter = t16_counter = t17_counter = t18_counter = t19_counter = t20_counter = t21_counter = t22_counter = t23_counter = t24_counter = 0
    t25_counter = t26_counter = t27_counter = t28_counter  = t29_counter = t30_counter = t31_counter = t32_counter = t33_counter = t34_counter = 0
    t35_counter = t36_counter = t37_counter = t38_counter = t39_counter = t40_counter = 0

    t1_counter_pos = t2_counter_pos = t3_counter_pos = t4_counter_pos = t5_counter_pos = t6_counter_pos = t7_counter_pos = t8_counter_pos = t9_counter_pos = t10_counter_pos = t11_counter_pos = t12_counter_pos = 0
    t13_counter_pos = t14_counter_pos = t15_counter_pos = t16_counter_pos = t17_counter_pos = t18_counter_pos = t19_counter_pos = t20_counter_pos = t21_counter_pos = t22_counter_pos = t23_counter_pos = t24_counter_pos = 0
    t25_counter_pos = t26_counter_pos = t27_counter_pos = t28_counter_pos  = t29_counter_pos = t30_counter_pos = t31_counter_pos = t32_counter_pos = t33_counter_pos = t34_counter_pos = 0
    t35_counter_pos = t36_counter_pos = t37_counter_pos = t38_counter_pos = t39_counter_pos = t40_counter_pos = 0

    t1_counter_neg = t2_counter_neg = t3_counter_neg = t4_counter_neg = t5_counter_neg = t6_counter_neg = t7_counter_neg = t8_counter_neg = t9_counter_neg = t10_counter_neg = t11_counter_neg = t12_counter_neg = 0
    t13_counter_neg = t14_counter_neg = t15_counter_neg = t16_counter_neg = t17_counter_neg = t18_counter_neg = t19_counter_neg = t20_counter_neg = t21_counter_neg = t22_counter_neg = t23_counter_neg = t24_counter_neg = 0
    t25_counter_neg = t26_counter_neg = t27_counter_neg = t28_counter_neg  = t29_counter_neg = t30_counter_neg = t31_counter_neg = t32_counter_neg = t33_counter_neg = t34_counter_neg = 0
    t35_counter_neg = t36_counter_neg = t37_counter_neg = t38_counter_neg = t39_counter_neg = t40_counter_neg = 0


    i = -1
    p = n = 0
    for sentence in X_full:
        i += 1
        if Y_full[i] == "1":
            p += 1
        else:
            n += 1
        if t1.match(sentence):
            t1_counter += 1
            if Y_full[i] == "1":
                t1_counter_pos += 1
            else:
                t1_counter_neg += 1
        elif t2.match(sentence):
            t2_counter += 1
            if Y_full[i] == "1":
                t2_counter_pos += 1
            else:
                t2_counter_neg += 1
        elif t3.match(sentence):
            t3_counter += 1
            if Y_full[i] == "1":
                t3_counter_pos += 1
            else:
                t3_counter_neg += 1
        elif t4.match(sentence):
            t4_counter += 1
            if Y_full[i] == "1":
                t4_counter_pos += 1
            else:
                t4_counter_neg += 1
        elif t5.match(sentence):
            t5_counter += 1
            if Y_full[i] == "1":
                t5_counter_pos += 1
            else:
                t5_counter_neg += 1
        elif t6.match(sentence):
            t6_counter += 1
            if Y_full[i] == "1":
                t6_counter_pos += 1
            else:
                t6_counter_neg += 1
        elif t7.match(sentence):
            t7_counter += 1
            if Y_full[i] == "1":
                t7_counter_pos += 1
            else:
                t7_counter_neg += 1
        elif t8.match(sentence):
            t8_counter += 1
            if Y_full[i] == "1":
                t8_counter_pos += 1
            else:
                t8_counter_neg += 1
        elif t9.match(sentence):
            t9_counter += 1
            if Y_full[i] == "1":
                t9_counter_pos += 1
            else:
                t9_counter_neg += 1
        elif t10.match(sentence):
            t10_counter += 1
            if Y_full[i] == "1":
                t10_counter_pos += 1
            else:
                t10_counter_neg += 1
        elif t11.match(sentence):
            t11_counter += 1
            if Y_full[i] == "1":
                t11_counter_pos += 1
            else:
                t11_counter_neg += 1
        elif t12.match(sentence):
            t12_counter += 1
            if Y_full[i] == "1":
                t12_counter_pos += 1
            else:
                t12_counter_neg += 1
        elif t13.match(sentence):
            t13_counter += 1
            if Y_full[i] == "1":
                t13_counter_pos += 1
            else:
                t13_counter_neg += 1
        elif t14.match(sentence):
            t14_counter += 1
            if Y_full[i] == "1":
                t14_counter_pos += 1
            else:
                t14_counter_neg += 1
        elif t15.match(sentence):
            t15_counter += 1
            if Y_full[i] == "1":
                t15_counter_pos += 1
            else:
                t15_counter_neg += 1
        elif t16.match(sentence):
            t16_counter += 1
            if Y_full[i] == "1":
                t16_counter_pos += 1
            else:
                t16_counter_neg += 1
        elif t17.match(sentence):
            t17_counter += 1
            if Y_full[i] == "1":
                t17_counter_pos += 1
            else:
                t17_counter_neg += 1
        elif t18.match(sentence):
            t18_counter += 1
            if Y_full[i] == "1":
                t18_counter_pos += 1
            else:
                t18_counter_neg += 1
        elif t19.match(sentence):
            t19_counter += 1
            if Y_full[i] == "1":
                t19_counter_pos += 1
            else:
                t19_counter_neg += 1
        elif t20.match(sentence):
            t20_counter += 1
            if Y_full[i] == "1":
                t20_counter_pos += 1
            else:
                t20_counter_neg += 1
        elif t21.match(sentence):
            t21_counter += 1
            if Y_full[i] == "1":
                t21_counter_pos += 1
            else:
                t21_counter_neg += 1
        elif t22.match(sentence):
            t22_counter += 1
            if Y_full[i] == "1":
                t22_counter_pos += 1
            else:
                t22_counter_neg += 1
        elif t23.match(sentence):
            t23_counter += 1
            if Y_full[i] == "1":
                t23_counter_pos += 1
            else:
                t23_counter_neg += 1
        elif t24.match(sentence):
            t24_counter += 1
            if Y_full[i] == "1":
                t24_counter_pos += 1
            else:
                t24_counter_neg += 1  
        elif t25.match(sentence):
            t25_counter += 1
            if Y_full[i] == "1":
                t25_counter_pos += 1
            else:
                t25_counter_neg += 1
        elif t26.match(sentence):
            t26_counter += 1
            if Y_full[i] == "1":
                t26_counter_pos += 1
            else:
                t26_counter_neg += 1
        elif t27.match(sentence):
            t27_counter += 1
            if Y_full[i] == "1":
                t27_counter_pos += 1
            else:
                t27_counter_neg += 1
        elif t28.match(sentence):
            t28_counter += 1
            if Y_full[i] == "1":
                t28_counter_pos += 1
            else:
                t28_counter_neg += 1
        elif t29.match(sentence):
            t29_counter += 1
            if Y_full[i] == "1":
                t29_counter_pos += 1
            else:
                t29_counter_neg += 1
        elif t30.match(sentence):
            t30_counter += 1
            if Y_full[i] == "1":
                t30_counter_pos += 1
            else:
                t30_counter_neg += 1
        elif t31.match(sentence):
            t31_counter += 1
            if Y_full[i] == "1":
                t31_counter_pos += 1
            else:
                t31_counter_neg += 1
        elif t32.match(sentence):
            t32_counter += 1
            if Y_full[i] == "1":
                t32_counter_pos += 1
            else:
                t32_counter_neg += 1
        elif t33.match(sentence):
            t33_counter += 1
            if Y_full[i] == "1":
                t33_counter_pos += 1
            else:
                t33_counter_neg += 1
        elif t34.match(sentence):
            t34_counter += 1 
            if Y_full[i] == "1":
                t34_counter_pos += 1
            else:
                t34_counter_neg += 1    
        elif t35.match(sentence):
            t35_counter += 1
            if Y_full[i] == "1":
                t35_counter_pos += 1
            else:
                t35_counter_neg += 1
        elif t36.match(sentence):
            t36_counter += 1
            if Y_full[i] == "1":
                t36_counter_pos += 1
            else:
                t36_counter_neg += 1
        elif t37.match(sentence):
            t37_counter += 1
            if Y_full[i] == "1":
                t37_counter_pos += 1
            else:
                t37_counter_neg += 1
        elif t38.match(sentence):
            t38_counter += 1
            if Y_full[i] == "1":
                t38_counter_pos += 1
            else:
                t38_counter_neg += 1
        elif t39.match(sentence):
            t39_counter += 1
            if Y_full[i] == "1":
                t39_counter_pos += 1
            else:
                t39_counter_neg += 1
        elif t40.match(sentence):
            t40_counter += 1
            if Y_full[i] == "1":
                t40_counter_pos += 1
            else:
                t40_counter_neg += 1
        else:
            print("Templates incomplete, there are still some templates not caught!")

    print(i)
    print(p)
    print(n)
    print("")      
    print("I like [word] , but not [word] . total:{0} pos: {1} neg: {2}".format(t1_counter, t1_counter_pos, t1_counter_neg))
    print("I like [word] , and [word] too . total:{0} pos: {1} neg: {2}".format(t2_counter, t2_counter_pos, t2_counter_neg))
    print("I like [word] more than [word] . total:{0} pos: {1} neg: {2}".format(t3_counter, t3_counter_pos, t3_counter_neg))
    print("I like [word] , an interesting type of [word] . total:{0} pos: {1} neg: {2}".format(t4_counter, t4_counter_pos, t4_counter_neg))
    print("I do not like [word] , I prefer [word] . total:{0} pos: {1} neg: {2}".format(t5_counter, t5_counter_pos, t5_counter_neg))
    print("I like [word] , and more specifically [word] . total:{0} pos: {1} neg: {2}".format(t6_counter, t6_counter_pos, t6_counter_neg))
    print("I use [word] , except [word] . total:{0} pos: {1} neg: {2}".format(t7_counter, t7_counter_pos, t7_counter_neg))
    print("He likes [word] , and ([word] too . total:{0} pos: {1} neg: {2}".format(t8_counter, t8_counter_pos, t8_counter_neg))
    print("I use [word] , and more specifically [word] . total:{0} pos: {1} neg: {2}".format(t9_counter, t9_counter_pos, t9_counter_neg))
    print("I met [word] , and more specifically [word] . total:{0} pos: {1} neg: {2}".format(t10_counter, t10_counter_pos, t10_counter_neg))
    print("He trusts his [word] , an interesting type of [word] . total:{0} pos: {1} neg: {2}".format(t11_counter, t11_counter_pos, t11_counter_neg))
    print("He trusts his [word] , except [word] . total:{0} pos: {1} neg: {2}".format(t12_counter, t12_counter_pos, t12_counter_neg))
    print("I like [word] , except [word] . total:{0} pos: {1} neg: {2}".format(t13_counter, t13_counter_pos, t13_counter_neg))
    print("I use [word] , and [word] too . total:{0} pos: {1} neg: {2}".format(t14_counter, t14_counter_pos, t14_counter_neg))
    print("He trusts [word] , and more specifically his [word] . total:{0} pos: {1} neg: {2}".format(t15_counter, t15_counter_pos, t15_counter_neg))
    print("He likes [word] , an interesting type of [word] . total:{0} pos: {1} neg: {2}".format(t16_counter, t16_counter_pos, t16_counter_neg))
    print("He does not trust [word] , he prefers his [word] . total:{0} pos: {1} neg: {2}".format(t17_counter, t17_counter_pos, t17_counter_neg))
    print("I met [word] , an interesting type of [word] . total:{0} pos: {1} neg: {2}".format(t18_counter, t18_counter_pos, t18_counter_neg))
    print("He does not like [word] , he prefers [word] . total:{0} pos: {1} neg: {2}".format(t19_counter, t19_counter_pos, t19_counter_neg))
    print("He trusts his [word] more than [word] . total:{0} pos: {1} neg: {2}".format(t20_counter, t20_counter_pos, t20_counter_neg))
    print("He trusts [word] , an interesting type of [word] . total:{0} pos: {1} neg: {2}".format(t21_counter, t21_counter_pos, t21_counter_neg))
    print("He trusts his [word] , but not [word] . total:{0} pos: {1} neg: {2}".format(t22_counter, t22_counter_pos, t22_counter_neg))
    print("I use [word] more than [word] . total:{0} pos: {1} neg: {2}".format(t23_counter, t23_counter_pos, t23_counter_neg))
    print("I use [word] , but not [word] . total:{0} pos: {1} neg: {2}".format(t24_counter, t24_counter_pos, t24_counter_neg))
    print("I met [word] , and [word] too . total:{0} pos: {1} neg: {2}".format(t25_counter, t25_counter_pos, t25_counter_neg))
    print("I use [word] , an interesting type of [word] . total:{0} pos: {1} neg: {2}".format(t26_counter, t26_counter_pos, t26_counter_neg))
    print("He likes [word] more than [word] . total:{0} pos: {1} neg: {2}".format(t27_counter, t27_counter_pos, t27_counter_neg))
    print("He does not trust his [word] , he prefers [word] . total:{0} pos: {1} neg: {2}".format(t28_counter, t28_counter_pos, t28_counter_neg))
    print("I met [word] , except [word] . total:{0} pos: {1} neg: {2}".format(t29_counter, t29_counter_pos, t29_counter_neg))
    print("I met [word] , but not [word] . total:{0} pos: {1} neg: {2}".format(t30_counter, t30_counter_pos, t30_counter_neg))
    print("He trusts his [word] , and [word] too . total:{0} pos: {1} neg: {2}".format(t31_counter, t31_counter_pos, t31_counter_neg))
    print("He likes [word] , but not [word] . total:{0} pos: {1} neg: {2}".format(t32_counter, t32_counter_pos, t32_counter_neg))
    print("He trusts his [word] , and more specifically [word] . total:{0} pos: {1} neg: {2}".format(t33_counter, t33_counter_pos, t33_counter_neg))
    print("He trusts [word] , except his [word] . total:{0} pos: {1} neg: {2}".format(t34_counter, t34_counter_pos, t34_counter_neg))
    print("He likes [word] , and more specifically [word] . total:{0} pos: {1} neg: {2}".format(t35_counter, t35_counter_pos, t35_counter_neg))
    print("He likes [word] , except [word] . total:{0} pos: {1} neg: {2}".format(t36_counter, t36_counter_pos, t36_counter_neg))
    print("He trusts [word] more than his [word] . total:{0} pos: {1} neg: {2}".format(t37_counter, t37_counter_pos, t37_counter_neg))
    print("He trusts [word] , but not his [word] . total:{0} pos: {1} neg: {2}".format(t38_counter, t38_counter_pos, t38_counter_neg))
    print("He trusts his [word] , and his [word] too . total:{0} pos: {1} neg: {2}".format(t39_counter, t39_counter_pos, t39_counter_neg))
    print("He trusts [word] , and his [word] too . total:{0} pos: {1} neg: {2}".format(t40_counter, t40_counter_pos, t40_counter_neg))
    print("")

if __name__ == "__main__":
    main()
