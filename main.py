import argparse

from config import Model_Config

'''
MATHia,  'Anon Student Id', 'KC (MATHia)', 'Correct'
ASSISTments, user_id, skill_id, correct

data_source: ASSISTments, DataShop

2012-2013-data-with-predictions-4-final.csv => ASSISTments => [user_id, problem_id, skill, correct, start_time]
algebra_2005_2006_train.txt, => DataShop =>[Anon Student Id, Problem Name, KC(Default), Correct First Attempt, First Transaction Time] 
bridge_to_algebra_2008_2009_train.txt => DataShop =>[Anon Student Id, Problem Name, KC(SubSkills), Correct First Attempt, First Transaction Time] 
'''

class KnowledgeTracing:
    def __init__(self):

        parser=argparse.ArgumentParser(description='Arguments for Parameters Setting')
        parser.add_argument("--data_path",nargs=1,type=str,default=['/dataset/2012-2013-data-with-predictions-4-final.csv'])
        parser.add_argument("--KT_model",nargs=1,type=str,default=['DKT'])
        parser.add_argument("--data_source",nargs=1,type=str,default=['ASSISTments'])
        parser.add_argument("--user_id",nargs=1,type=str,default=['user_id'])
        parser.add_argument("--problem_id",nargs=1,type=str,default=['problem_id'])
        parser.add_argument("--skill_id",nargs=1,type=str,default=['skill'])
        parser.add_argument("--correct",nargs=1,type=str,default=['correct'])
        parser.add_argument("--timestamp",nargs=1,type=str,default=['start_time'])
        
        args=parser.parse_args()
        self.args=args
        
    def main(self):
        
        print("Knowledge Tracing Demo")
        
        config=Model_Config(self.args)
        config.main()
        
if __name__ == '__main__':
    print("main")
    obj=KnowledgeTracing()
    obj.main()