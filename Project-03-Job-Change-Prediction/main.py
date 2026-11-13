from src.utils.logger import setup_logger

from src.preprocessing.data_cleaning import run_data_cleaning
from src.preprocessing.prepare_data import run_preprocessing

from src.models.train_model import run_training
from src.models.evaluate_model import run_evaluation
from src.models.imbalance_experiments import run_experiments
from src.models.regularization import run_regularization
from src.models.cross_validation import run_cross_validation
from src.models.hyperparameter_tuning import run_hyperparameter_tuning
from src.models.final_evaluation import run_final_evaluation


# =====================================================
# 1. LOGGER
# =====================================================

logger = setup_logger()


# =====================================================
# 2. PIPELINE STAGE EXECUTOR
# =====================================================

def execute_stage(stage_name, stage_function):

    logger.info("=" * 60)
    logger.info(f"STARTING: {stage_name}")
    logger.info("=" * 60)

    stage_function()

    logger.info(f"COMPLETED: {stage_name}")


# =====================================================
# 3. MAIN PIPELINE
# =====================================================

def main():

    logger.info("=" * 60)
    logger.info("JOB CHANGE PREDICTION")
    logger.info("END-TO-END MACHINE LEARNING PIPELINE")
    logger.info("=" * 60)

    try:

        # STEP 1: DATA CLEANING

        execute_stage(
            "DATA CLEANING",
            run_data_cleaning
        )

        # STEP 2: PREPROCESSING

        execute_stage(
            "DATA PREPROCESSING",
            run_preprocessing
        )

        # STEP 3: BASELINE MODEL TRAINING

        execute_stage(
            "BASELINE MODEL TRAINING",
            run_training
        )

        # STEP 4: BASELINE MODEL EVALUATION

        execute_stage(
            "BASELINE MODEL EVALUATION",
            run_evaluation
        )

        # STEP 5: IMBALANCE EXPERIMENTS

        execute_stage(
            "CLASS IMBALANCE EXPERIMENTS",
            run_experiments
        )

        # STEP 6: REGULARIZATION

        execute_stage(
            "REGULARIZATION EXPERIMENTS",
            run_regularization
        )

        # STEP 7: CROSS VALIDATION

        execute_stage(
            "CROSS VALIDATION",
            run_cross_validation
        )

        # STEP 8: HYPERPARAMETER TUNING

        execute_stage(
            "HYPERPARAMETER TUNING",
            run_hyperparameter_tuning
        )

        # STEP 9: FINAL MODEL EVALUATION

        execute_stage(
            "FINAL MODEL EVALUATION",
            run_final_evaluation
        )

        logger.info("=" * 60)
        logger.info("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        print("\n============================================")
        print("JOB CHANGE PREDICTION PIPELINE COMPLETED")
        print("============================================")

    except Exception:

        logger.exception(
            "MACHINE LEARNING PIPELINE FAILED"
        )

        raise


# =====================================================
# 4. EXECUTION
# =====================================================

if __name__ == "__main__":

    main()