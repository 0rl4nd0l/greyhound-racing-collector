"""Scientific seams: as-of histories, layout comparability and nested selection."""
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
import numpy as np
from scripts import offline_systematic_features as features
from scripts import offline_systematic_search as search
from scripts.offline_prediction_research import fit_linear,predict_linear

class SystematicTests(unittest.TestCase):
    def history(self):
        return [{'DATE':f'2026-06-{i:02d}','TRACK':'BEN','DIST':'400','G':'5','PLC':str(i-4),'MGN':str(i),'BOX':'1','TIME':str(25+i/10),'1 SEC':str(4+i/100)} for i in [8,7,6]]
    def rows(self):
        return [{'race_id':f'Race {j}','race_date':f'2026-06-{10+j:02d}','box':i+1,'layout_venue':'BEN','y':int(i==j%3),'market':[.5,.3,.2][i],'features':{'x':float(i-j),'box_fraction':i/7}} for j in range(8) for i in range(3)]
    def test_history_future_rows_cannot_enter_raw_extension(self):
        rows=self.history()+[{'DATE':'2026-06-10','PLC':'1'}]
        with self.assertRaisesRegex(ValueError,'membership'):
            features.history_features(rows,date(2026,6,10),'BEN',400,'5')
    def test_times_use_same_layout_and_distance_only(self):
        raw=self.history()
        a=features.history_features(raw,date(2026,6,10),'BEN',400,'5')
        self.assertEqual(a['same_layout_time_count'],3)
        b=features.history_features(raw,date(2026,6,10),'BEN',500,'5')
        self.assertEqual(b['same_layout_time_count'],0);self.assertIsNone(b['_time_mean'])
        for r in raw:r['TRACK']='QOT'
        c=features.history_features(raw,date(2026,6,10),'QOT',400,'5')
        self.assertEqual(c['same_layout_time_count'],0)
    def test_single_digit_pir_is_not_a_feature(self):
        raw=self.history();a=features.history_features(raw,date(2026,6,10),'BEN',400,'5')
        for r in raw:r['PIR']='12345678'
        self.assertEqual(a,features.history_features(raw,date(2026,6,10),'BEN',400,'5'))
    def test_new_vectorized_method_reproduces_reference(self):
        rows=self.rows();old=fit_linear(rows,['x']);new=search.linear_fit(rows,['x'],1)
        np.testing.assert_allclose(search.predict(rows,new),predict_linear(rows,old),atol=1e-9)
    def test_noncontiguous_race_rejected(self):
        rows=self.rows();rows.append(rows[0])
        with self.assertRaisesRegex(ValueError,'contiguous'):search.starts(rows)
    def test_new_venue_is_not_fitted_from_later_data(self):
        rows=self.rows();prep=search.prep_fit(rows,['x','__box_track__'])
        self.assertEqual([n for n in prep['names'] if n.startswith('venue_box')],['venue_box::BEN'])
        future=[{**rows[0],'layout_venue':'NEW','features':{'x':1e9,'box_fraction':1}}]
        old=prep['mean'].copy();search.transform(future,prep)
        np.testing.assert_array_equal(old,prep['mean'])
    def test_all_nested_splits_precede_outer_dates(self):
        for outer in search.OUTER:
            for end,start,finish in outer['inner']:
                self.assertLess(end,start);self.assertLess(finish,outer['test_start'])
    def test_ledger_is_append_only_and_hash_chain_verifies(self):
        import hashlib
        with tempfile.TemporaryDirectory() as path:
            p=Path(path)/'ledger';l=search.Ledger(p);l.append('START',variant='bad');l.append('FAILED',reason='test')
            previous='0'*64
            for line in p.read_text().splitlines():
                r=json.loads(line);digest=r.pop('sha256');self.assertEqual(r['previous_sha256'],previous)
                self.assertEqual(digest,hashlib.sha256(search.canonical(r).encode()).hexdigest());previous=digest
            with self.assertRaises(ValueError):search.Ledger(p)
    def test_no_qualified_rule_means_abstain(self):
        rows=self.rows()
        for r in rows:
            r['odds']=1/r['market'];r['features'].update(limited_history=0,layoff_21=0,faster_neighbors=None,margin_recent_long_gap=1,time_field_rank=None)
        with tempfile.TemporaryDirectory() as path:
            l=search.Ledger(Path(path)/'ledger');p=np.array([r['market'] for r in rows]);rules=search.select_rules(rows,p,l,'synthetic')
            self.assertEqual(rules,{'favorite':None,'outsider':None})

if __name__=='__main__':unittest.main()
