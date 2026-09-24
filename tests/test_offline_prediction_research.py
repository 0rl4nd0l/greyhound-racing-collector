"""Small synthetic tests of the new evaluation seam, without real outcomes."""
import unittest
import numpy as np
from scripts import offline_prediction_research as r

class ResearchTests(unittest.TestCase):
 def setUp(self):
  self.rows=[dict(race_id=f'race{j}',race_date=f'2026-06-{10+j:02}',y=int(i==j%3),market=[.5,.3,.2][i],features={'x':float(i-j)}) for j in range(8) for i in range(3)]
 def test_normalization_and_shift(self):
  z=np.arange(24)/10
  a=r.probabilities(self.rows,z);b=r.probabilities(self.rows,z+100)
  np.testing.assert_allclose(a,b,atol=1e-14)
  for start,end in zip(*r.group(self.rows)):self.assertAlmostEqual(a[start:end].sum(),1)
 def test_train_only_preprocessing_and_missing_indicator(self):
  train=self.rows[:3];prep=r.prefit(train,['x']);old=prep['mean'].copy()
  r.transform([dict(race_id='later',features={'x':99999}),dict(race_id='later',features={'x':None})],prep)
  np.testing.assert_array_equal(old,prep['mean']);self.assertEqual(prep['med'][0],1)
 def test_conditional_model_is_row_permutation_invariant(self):
  model=r.fit_linear(self.rows,['x']);p=r.predict_linear(self.rows,model)
  swapped=[];expected=[]
  for start,end in zip(*r.group(self.rows)):
   swapped.extend(self.rows[start:end][::-1]);expected.extend(p[start:end][::-1])
  np.testing.assert_allclose(r.predict_linear(swapped,model),expected,atol=1e-12)
 def test_penalty_and_cap_match_frozen_method(self):
  model=r.fit_linear(self.rows,['x']);beta=model['beta'];x=r.transform(self.rows,model['prep']);y=np.array([v['y'] for v in self.rows]);base=np.log([v['market'] for v in self.rows]);n=8
  def objective(b):
   p=r.probabilities(self.rows,base+.35*np.tanh(x@b/.35));return -np.log(p[y==1]).mean()+.5*np.dot(b,b)
  eps=1e-5
  gradient=np.array([(objective(beta+eps*np.eye(len(beta))[i])-objective(beta-eps*np.eye(len(beta))[i]))/(2*eps) for i in range(len(beta))])
  self.assertLess(np.linalg.norm(gradient),1e-5)
  ratio=r.predict_linear(self.rows,model)/np.exp(base)
  for a,b in zip(*r.group(self.rows)):self.assertLessEqual(np.log(ratio[a:b]).max()-np.log(ratio[a:b]).min(),.7+1e-12)
 def test_chronological_folds_and_unique_tests(self):
  previous=''
  for f in r.FOLDS:
   self.assertLess(f['train_end'],f['val_start']);self.assertLess(f['val_end'],f['test_start']);self.assertLess(previous,f['test_start']);previous=f['test_end']
 def test_label_not_needed_for_identity_projection(self):
  line='{"race_id":"Race 1 - BEN - 2026-07-17", "label": UNDECODABLE}'
  self.assertEqual(r.scalar(line,'race_id'),'Race 1 - BEN - 2026-07-17')
  self.assertEqual(r.race_key(r.scalar(line,'race_id')),'2026-07-17|BEN|1')

if __name__=='__main__':unittest.main()
