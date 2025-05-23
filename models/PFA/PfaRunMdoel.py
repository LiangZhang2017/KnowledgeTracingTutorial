
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# import statsmodels.api as sm
# from pymer4.models import Lmer
# from sklmer import LmerRegressor

from PFA.pfa import PFA
from PFA.PFA_helper import CheckKCMatch

def PFAModel(PFA_config):
    exper_data=PFA_config['exper_data']
    cvt=PFA_config['cvt']
    CV=PFA_config['cvfold']
    is_cross_validation = PFA_config['is_cross_validation']

    PFAmodel_Config={
        'metrics':PFA_config['metrics'],
        'flags':PFA_config['flags'],
        'KCmodel':PFA_config['KCmodel'],
        'KCs': len(set(exper_data['KC_Theoretical_Levels'])),
        'KC_used': 'KC_Theoretical_Levels',
        'is_cross_validation':PFA_config['is_cross_validation'],
        'ComputeFeatures':PFA_config['ComputeFeatures'],
        'CV':PFA_config['cvfold'],
        'exper_data':exper_data
    }

    if is_cross_validation is True:
        mae_list=[]
        rmse_list=[]
        auc_list=[]

        for cv in range(1, cvt + 1):
            studentsIds = np.unique(exper_data['Student_Id'])
            kf = KFold(n_splits=CV, random_state=cv, shuffle=True)
            for k, (train, test) in enumerate(kf.split(studentsIds)):
                print("The kth fold is {}".format(k))
                print("The train:test is {}:{}".format(len(train),len(test)))
                train_data = exper_data.loc[exper_data['Student_Id'].isin(studentsIds[train])]
                test_data = exper_data.loc[exper_data['Student_Id'].isin(studentsIds[test])]

                train_data_check,test_data_check=CheckKCMatch(train_data,test_data,PFAmodel_Config['KC_used'])

                PFAmodel_Config['train_data']=train_data_check
                PFAmodel_Config['test_data']=test_data_check
                model=PFA(PFAmodel_Config)
                model.training()

                print(model.metrics)

                mae_list.append(model.metrics[0])
                rmse_list.append(model.metrics[1])
                auc_list.append(model.metrics[2])

        return mae_list, rmse_list, auc_list

    if is_cross_validation is False:
        mae_list=[]
        rmse_list=[]
        auc_list=[]

        for cv in range(1, cvt + 1):
            PFAmodel_Config['train_data'] =exper_data
            PFAmodel_Config['test_data'] = None
            model = PFA(PFAmodel_Config)
            model.training()
            print("The cv is {}".format(cv))
            print(model.metrics)
            mae_list.append(model.metrics[0])
            rmse_list.append(model.metrics[1])
            auc_list.append(model.metrics[2])

        return mae_list,rmse_list,auc_list