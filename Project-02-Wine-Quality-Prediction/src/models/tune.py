from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC


def tune_models(X_train, y_train):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    models = {

        "Random Forest": {

            "model": RandomForestClassifier(
                class_weight="balanced",
                random_state=42
            ),

            "params": {

                "n_estimators": [100, 200],

                "max_depth": [None, 10, 20],

                "min_samples_leaf": [1, 2, 4]

            }
        },

        "SVM": {

            "model": Pipeline([

                ("scaler", StandardScaler()),

                ("model", SVC(
                    class_weight="balanced"
                ))

            ]),

            "params": {

                "model__C": [0.1, 1, 10],

                "model__kernel": ["linear", "rbf"],

                "model__gamma": ["scale", "auto"]

            }
        }

    }

    best_models = {}

    for name, config in models.items():

        print(f"\nTuning {name}...")

        grid = GridSearchCV(

            estimator=config["model"],

            param_grid=config["params"],

            scoring="f1",

            cv=cv,

            n_jobs=-1

        )

        grid.fit(X_train, y_train)

        best_models[name] = grid.best_estimator_

        print("\nBest Parameters:")
        print(grid.best_params_)

        print("Best CV F1 Score:")
        print(grid.best_score_)

    return best_models