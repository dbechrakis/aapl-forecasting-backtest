import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coursework"))
from main import choose_ses_alpha, choose_ma_window, naive_forecast, moving_average_forecast, ses_forecast  # noqa: E402

class ForecastingTests(unittest.TestCase):
    def test_ses_uses_prior_level(self):
        # Alternating observations favour smoothing; fitting each observation first
        # mechanically favours the largest alpha and fails this check.
        train=np.array([10.,0.,10.,0.,10.,0.])
        candidates=[.1,.5,.9]
        losses=[]
        for alpha in candidates:
            prior=train[0]
            loss=0.
            for observed in train[1:]:
                loss+=abs(observed-prior)
                prior=alpha*observed+(1-alpha)*prior
            losses.append(loss)
        self.assertEqual(choose_ses_alpha(train,candidates),candidates[int(np.argmin(losses))])
        self.assertNotEqual(choose_ses_alpha(train,candidates),.9)

    def test_ma_common_validation_dates(self):
        train=np.array([1.,10.,2.,9.,3.,8.,4.,7.,5.,6.,15.])
        candidates=[2,3,5]
        errors={w:sum(abs(train[i]-sum(train[i-w:i])/w) for i in range(5,len(train))) for w in candidates}
        self.assertEqual(choose_ma_window(train,candidates),min(errors,key=errors.get))

    def test_constant_baselines_and_horizon(self):
        train=np.full(12,7.)
        for result in [naive_forecast(train,5),moving_average_forecast(train,5,3),ses_forecast(train,5,.3)]:
            np.testing.assert_allclose(result,np.full(5,7.))

if __name__=="__main__":
    unittest.main()
