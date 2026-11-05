import unittest
import numpy as np
from pcf import PCF
from pcf.io import split_indices

class PCFIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(17)
        cls.p=rng.dirichlet(np.ones(4),size=(240,5))
        cls.original=cls.p.copy()
        cls.model=PCF().fit(cls.p[:200])

    def test_heldout_probabilities_and_batch_invariance(self):
        q=self.model.predict_proba(self.p[200:])
        self.assertEqual(q.shape,(40,4))
        self.assertTrue(np.isfinite(q).all())
        np.testing.assert_allclose(q.sum(1),1,atol=1e-6)
        np.testing.assert_allclose(q[:7],self.model.predict_proba(self.p[200:207]),atol=1e-6)
        np.testing.assert_array_equal(self.p,self.original)

    def test_paper_split_has_no_overlap_and_preserves_order(self):
        tr,te=split_indices(101,2026)
        expected=np.random.default_rng(2026).permutation(101)
        np.testing.assert_array_equal(te,np.sort(expected[:20]))
        np.testing.assert_array_equal(tr,np.sort(expected[20:]))
        self.assertFalse(np.intersect1d(tr,te).size)

    def test_invalid_inputs_and_unfitted_prediction(self):
        with self.assertRaises(RuntimeError):PCF().predict_proba(self.p)
        for bad in [np.zeros((3,5,4)),np.full((3,5,4),np.nan),-self.p[:3],np.ones((0,5,4))]:
            with self.assertRaises(ValueError):PCF().fit(bad)
        with self.assertRaises(ValueError):self.model.predict_proba(self.p[:3,:,:3])

if __name__=='__main__':unittest.main()
