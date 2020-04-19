import pandas as pd

submission_0 = pd.read_csv("submission_1.csv")
submission_1 = pd.read_csv("submission_1.csv")

submission = submission_0.copy()
submission[['healthy', 'multiple_diseases', 'rust', 'scab']] = (submission_0[['healthy', 'multiple_diseases', 'rust', 'scab']]
                                                                + submission_1[['healthy', 'multiple_diseases', 'rust', 'scab']]) / 2
submission.to_csv("submission.csv", index=False)