# pfa_full.py  -----------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from pymer4.models import Lmer

# --- R ↔ Python bridges -----------------------------------------------
pandas2ri.activate()
numpy2ri.activate()

lme4  = importr("lme4")
stats = importr("stats")

# ----------------------------------------------------------------------
class PFA:
    """
    Performance-Factors Analysis (kc_cor / kc_incor).
    flags = "Simple" (global slopes) | "Full" (global + KC-specific slopes)
    """

    def __init__(self, config: dict):
        self.flags        = config.get("flags", "Simple")
        self.KC_used      = config["KC_used"]            # e.g. "skill_name"
        self.USER_used    = config.get("USER_used", "user_id")
        self.train_data   = config["train_data"].copy()
        self.test_data    = config["test_data"].copy()
        self.scale_pred   = config.get("scale_predictors", False)
        self.metrics      = {}

        # --- Internal hyperparameters (can be fixed here) --------------
        self.nAGQ         = 0
        self.optimizer    = "bobyqa"
        self.maxfun       = 200000
        self.tolPwrss     = 1e-5
        self.succ_scale   = 1.0      # scale factor for kc_cor
        self.fail_scale   = 1.0      # scale factor for kc_incor
        self.gamma_fixed  = None     # fixed success weight (None = estimate)
        self.rho_fixed    = None     # fixed failure weight (None = estimate)

        if self.scale_pred:
            for col, lam in zip(["kc_cor", "kc_incor"], [self.succ_scale, self.fail_scale]):
                self.train_data[col] = StandardScaler().fit_transform(self.train_data[[col]]) * lam
                self.test_data[col]  = StandardScaler().fit_transform(self.test_data[[col]]) * lam

        if self.gamma_fixed is not None and self.rho_fixed is not None:
            self.train_data["pf_offset"] = (
                self.gamma_fixed * self.train_data["kc_cor"] +
                self.rho_fixed   * self.train_data["kc_incor"]
            )
            self.test_data["pf_offset"] = (
                self.gamma_fixed * self.test_data["kc_cor"] +
                self.rho_fixed   * self.test_data["kc_incor"]
            )

    # ------------------------------------------------------------------
    def _build_formula(self) -> str:
        kc, user = self.KC_used, self.USER_used
        if self.gamma_fixed is not None and self.rho_fixed is not None:
            return f"score ~ offset(pf_offset) + (1|{kc}) + (1|{user})"
        if self.flags.lower() == "simple":
            return f"score ~ kc_cor + kc_incor + (1|{kc}) + (1|{user})"
        return (
            f"score ~ kc_cor + kc_incor "
            f"+ kc_cor:{kc} + kc_incor:{kc} "
            f"+ (1|{kc}) + (1|{user})"
        )

    # ------------------------------------------------------------------
    def _align_levels(self):
        """Ensure train & test share identical categorical levels."""
        kc_levels   = pd.Categorical(self.train_data[self.KC_used]).categories
        user_levels = pd.Categorical(self.train_data[self.USER_used]).categories
        for df in (self.train_data, self.test_data):
            df[self.KC_used]   = pd.Categorical(df[self.KC_used],  categories=kc_levels)
            df[self.USER_used] = pd.Categorical(df[self.USER_used], categories=user_levels)

    # ------------------------------------------------------------------
    @staticmethod
    def _r_predict():
        """
        Custom R function:
        1) builds model matrix for newdata,
        2) keeps only columns present in fixef(),
        3) returns plogis(Xβ)   (random effects = 0 for unseen levels).
        """
        return ro.r("""
            function(model, newdata) {
              fe  <- lme4::fixef(model)
              mm  <- model.matrix(stats::delete.response(stats::terms(model)), newdata)
              keep <- intersect(colnames(mm), names(fe))
              mm   <- mm[ , keep, drop = FALSE]
              eta  <- mm %*% fe[keep]
              stats::plogis(eta)
            }
        """)

    # ------------------------------------------------------------------
    def training(self):
        self._align_levels()                       # ---- level harmonisation
        formula = self._build_formula()
        print("⏳  Fitting GLMM with formula:\n   ", formula)

        ctrl_code = (
            f'lme4::glmerControl(optimizer = "{self.optimizer}", '
            f'optCtrl = list(maxfun = {self.maxfun}), '
            f'tolPwrss = {self.tolPwrss})'
        )
        ctrl = ro.r(ctrl_code)

        try:
            model = lme4.glmer(
                formula = formula,
                data    = self.train_data,
                family  = stats.binomial,
                control = ctrl,
                nAGQ    = self.nAGQ                    # Laplace approx.
            )
            print("✅  glmer converged.")
        except ro.rinterface_lib.embedded.RRuntimeError:
            print("⚠️  glmer failed; falling back to pymer4.Lmer.")
            model = Lmer(formula, data=self.train_data, family="binomial")
            model.fit(summarize=False)

        preds = np.asarray(self._r_predict()(model, self.test_data))
        obs   = self.test_data["score"].to_numpy()
        self.metrics = self.evaluate(preds, obs)
        print("📊  Metrics:", self.metrics)

    # ------------------------------------------------------------------
    @staticmethod
    def evaluate(pred, obs):
        pred = np.asarray(pred).ravel()
        obs  = np.asarray(obs).ravel().astype(int)

        mae  = mean_absolute_error(obs, pred)
        try:
            rmse = mean_squared_error(obs, pred, squared=False)
        except TypeError:                          # older sklearn
            rmse = np.sqrt(mean_squared_error(obs, pred))

        auc  = roc_auc_score(obs, pred) if len(np.unique(obs)) == 2 else np.nan
        return {"MAE": mae, "RMSE": rmse, "AUC": auc}