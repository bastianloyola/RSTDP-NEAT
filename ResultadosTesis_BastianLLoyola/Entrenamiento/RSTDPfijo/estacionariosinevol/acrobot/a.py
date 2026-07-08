import optuna

DB_PATH = "example-study.db"   # path a tu DB
STUDY_NAME = None              # o el nombre del study

study = optuna.load_study(
    study_name=STUDY_NAME,
    storage=f"sqlite:///{DB_PATH}"
)

best_trial = study.best_trial

print(f"Best value: {best_trial.value}")
print(f"Trial number: {best_trial.number}")
for k, v in best_trial.params.items():
    print(f"{k}={v}")
