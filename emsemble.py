import pandas as pd

submission_0 = pd.read_csv("submission_0.csv")
submission_1 = pd.read_csv("submission_1.csv")
submission_2 = pd.read_csv("submission_2.csv")
submission_3 = pd.read_csv("submission_3.csv")
submission_4 = pd.read_csv("submission_4.csv")

submission = submission_0.copy()
submission[['healthy', 'multiple_diseases', 'rust', 'scab']] = \
    (submission_0[['healthy', 'multiple_diseases', 'rust', 'scab']]
     + submission_1[['healthy', 'multiple_diseases', 'rust', 'scab']]
     + submission_2[['healthy', 'multiple_diseases', 'rust', 'scab']]
     + submission_3[['healthy', 'multiple_diseases', 'rust', 'scab']]
     + submission_4[['healthy', 'multiple_diseases', 'rust', 'scab']]) / 5

submission.to_csv("submission.csv", index=False)