import json
import unittest
import numpy as np
from scripts.explain_market_residual import decomposition, gate_lines, losses, bootstrap_tables


class ExplanationTests(unittest.TestCase):
    def test_proper_score_and_ties_are_distinct(self):
        a=losses([1,0],[.5,.5]); b=losses([1,0],[.6,.4])
        self.assertEqual(a['accuracy'],.5)
        self.assertAlmostEqual(a['brier'],.5)
        self.assertGreater(a['ll'],b['ll'])
        with self.assertRaises(ValueError): losses([1,1],[.5,.5])
        with self.assertRaises(ValueError): losses([1,0],[.5,.6])

    def test_identity_gate_rejects_before_full_decode(self):
        line='{"race_id":"Race 1 - X - 2026-08-01","box":1,"dog_token":"X","y": THIS_IS_NOT_JSON}'
        with self.assertRaisesRegex(ValueError,'unadmitted'):
            gate_lines(line.encode(),{})

    def test_incomplete_and_duplicate_rosters_rejected(self):
        rid='Race 1 - X - 2026-06-10'
        row={'race_id':rid,'race_date':'2026-06-10','box':1,'dog_token':'X','y':1}
        allowed={(rid,1):'X',(rid,2):'Y'}
        payload=json.dumps(row).encode()
        with self.assertRaisesRegex(ValueError,'incomplete'): gate_lines(payload,allowed)
        with self.assertRaisesRegex(ValueError,'duplicate'): gate_lines(payload+b'\n'+payload,allowed)

    def test_decomposition_includes_imputation_cap_and_normalization(self):
        rows=[{'features':{'f':None},'market':.7},{'features':{'f':4},'market':.3}]
        model={'prep':{'names':['f'],'median':[2],'mean':[3,.5],'scale':[1,.5],'center':True},'beta':[.4,.2]}
        d=decomposition(rows,model)
        np.testing.assert_allclose(d['contributions'].sum(1),d['z'])
        np.testing.assert_allclose(d['z'],[-.2,.2])
        self.assertNotEqual(d['log_normalizer'],0)
        np.testing.assert_allclose(np.log(d['p']/[.7,.3]),d['cap']-d['log_normalizer'])
        self.assertAlmostEqual(d['p'].sum(),1)
        self.assertLess(abs(d['cap'][0]),abs(d['z'][0]))

    def test_date_clusters_and_lodo_use_race_weighting(self):
        rows=[]
        for day,gain in [('a',.1),('a',.1),('b',-.2)]:
            row={'race_date':day,'groups':{'one':'all'},'market':{'ll':1.,'brier':1.,'accuracy':.5,'top_ties':1}}
            for model in ['refit_base16','refit_half','refit_box']:
                row[model]={'ll':1-gain,'brier':1-gain,'accuracy':.5}
            rows.append(row)
        table,meta=bootstrap_tables(rows,200)
        self.assertAlmostEqual(table[0]['ll']['improvement'],0)
        np.testing.assert_allclose(table[0]['ll']['lodo_range'],[-.2,.1])
        self.assertEqual(meta['dates'],['a','b'])
        self.assertEqual(table[0]['races'],3)
        self.assertEqual(table[0]['dates'],2)

if __name__=='__main__': unittest.main()
