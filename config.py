import os
import pandas as pd
from sklearn.model_selection import KFold
from models.bkt import Model as bkt_model
from models.PFA.pfa import PFA
from models.PFA.PFA_helper import add_pfa_features
from models.DKT.dkt import DKT
from models.DKT.dkt_helper import make_loaders
import numpy as np
from sklearn.model_selection import GroupKFold   # replaces plain KFold
from sklearn.metrics import mean_absolute_error
from pyBKT.util import metrics

metrics.SUPPORTED_METRICS.setdefault("mae", mean_absolute_error)

import torch
from EduData import get_data

class Model_Config:
    def __init__(self,args):
        self.args=args
        
    def main(self):
        print("start run model")
        
        #Read the dataset from file path
        full_path = os.getcwd()+self.args.data_path[0]
        df = pd.read_csv(full_path)
        print(df.columns)
        
        TARGET_N = 500
        RANDOM_SEED = 24
        
        df_new = pd.DataFrame(index=df.index) 

        # Mapping columns
        df_new['user_id']=df[self.args.user_id[0]]
        df_new['skill_name']=df[self.args.skill_id[0]]
        df_new['correct']=df[self.args.correct[0]].astype(int)
        df_new['timestamp']=df[self.args.timestamp[0]]
        df_new['problem_id']=df[self.args.problem_id[0]]
        
        df_new = df_new.dropna(subset=['skill_name'])
        df_new = df_new.dropna(subset=['correct'])
        
        # 1. keep earliest try per learner–problem  (skill_name optional here)
        df_new = (df_new
            .groupby(["user_id", "skill_name"], as_index=False, sort=False)
            .first())
                
        unique_users = df_new['user_id'].unique()   
        n_users = len(unique_users)                 

        if n_users > TARGET_N:                      
            rng = np.random.default_rng(RANDOM_SEED)
            sampled_users = rng.choice(unique_users,
                                    size=TARGET_N,
                                    replace=False)
            df_new = df_new[df_new['user_id'].isin(sampled_users)]
            print(f"Sub-sampled to {TARGET_N} students "
                f"({len(df_new)} total interactions).")
        else:
            print(f"Dataset has only {n_users} students – using them all.")
    
        print(df_new.head())
        
        unique_users = df_new["user_id"].unique()
        print("unique users:", len(unique_users))
        
        unique_questions = df_new['problem_id'].unique()   
        print("unique questions:", len(unique_questions))
        
        unique_skills = df_new['skill_name'].unique()   
        print("unique skills:", len(unique_skills))
        
        
        metrics = {"MAE": [], "RMSE": [], "AUC": []}
            
        # ----- 5-fold, user-level CV ------------------------------------------
        gkf = GroupKFold(n_splits=5)                     
        
        fold = 1
        
        for train_idx, test_idx in gkf.split(unique_users, groups=unique_users):
            print(f"\nFold {fold}")

            train_users = unique_users[train_idx]
            test_users  = unique_users[test_idx]

            train_df = df_new[df_new["user_id"].isin(train_users)].copy()
            test_df  = df_new[df_new["user_id"].isin(test_users)].copy()

            print(f"Train users: {len(train_users)}, Test users: {len(test_users)}")
            print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            
            if self.args.KT_model[0]=='BKT':
               print("Standard setting")

               model = bkt_model(seed=42, num_fits=5)
               model.fit(data=train_df)
               training_rmse = model.evaluate(data = train_df)
               print("training_rmse is ", training_rmse)
               training_auc = model.evaluate(data = train_df, metric = 'auc')
               print("training_auc is ", training_auc)

               # predictions = model.predict(data=test_df)

               # ---- training metrics -------------------------------------
               training_rmse = model.evaluate(data=train_df)                 # default is rmse
               training_auc  = model.evaluate(data=train_df,  metric="auc")
               training_mae  = model.evaluate(data=train_df,  metric="mae")  # now works

               print("training_rmse:", training_rmse)
               print("training_auc :", training_auc)
               print("training_mae :", training_mae)

                # ---- test metrics -----------------------------------------
               test_rmse = model.evaluate(data=test_df)                      # rmse
               test_auc  = model.evaluate(data=test_df, metric="auc")
               test_mae  = model.evaluate(data=test_df, metric="mae")        # mae
               
               print("test_rmse   :", test_rmse)
               print("test_auc    :", test_auc)
               print("test_mae    :", test_mae)

            if self.args.KT_model[0]=='PFA':
               print("run PFA")
               
               train_df = add_pfa_features(train_df)
               test_df  = add_pfa_features(test_df)
               
               pfa_cfg = {
                    "KC_used":  "skill_name",
                    "train_data": train_df,
                    "test_data":  test_df,
                    }
               
               model = PFA(pfa_cfg)
               model.training()
               
               test_mae  = model.metrics["MAE"]
               test_rmse = model.metrics["RMSE"]
               test_auc  = model.metrics["AUC"]
               
            if self.args.KT_model[0] == 'DKT':
                print("DKT")

                DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                BATCH_SIZE = 20
                EPOCHS     = 100
                HIDDEN     = 100

                # ------------------------------------------------------------
                # 1) build DataLoaders
                # ------------------------------------------------------------
                train_loader, test_loader, n_skill = make_loaders(train_df, test_df)

                # ------------------------------------------------------------
                # 2) create the DKT wrapper (CPU; its inner .dkt_model is PyTorch)
                # ------------------------------------------------------------
                model = DKT(
                    num_questions=n_skill,
                    hidden_size=32,
                    num_layers=3,     
                    dropout=0.4       
                )

                # (optional) move the inner net to GPU if available
                model.dkt_model.to(DEVICE)

                # ------------------------------------------------------------
                # 3) train — your DKT class already implements its own loop
                # ------------------------------------------------------------
                model.train(
                    train_loader,
                    test_data=test_loader,
                    epoch=100,
                    lr=3e-4,          # >100× larger
                    weight_decay=1e-4
                )

                # ------------------------------------------------------------
                # 4) final evaluation
                # ------------------------------------------------------------
                test_mae, test_rmse, test_auc = model.eval(test_loader)
            
            for k, v in zip(["MAE", "RMSE", "AUC"], [test_mae, test_rmse, test_auc]):
                metrics[k].append(v)

            print(f"Fold {fold} | "
                f"RMSE={test_rmse:.4f}  AUC={test_auc:.4f}  MAE={test_mae:.4f}")

            fold += 1
        
        print("\n==== 5-Fold Summary ====")
        for k, vals in metrics.items():
            mean_val  = np.mean(vals)
            fold_vals = "  ".join(f"{v:.4f}" for v in vals)
            print(f"{k}: {fold_vals}  |  mean = {mean_val:.4f}")