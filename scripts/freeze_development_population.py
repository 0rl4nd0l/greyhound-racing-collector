"""Default-off read of the existing collector index at the fixed pilot cutoff."""
import argparse
import json
from pathlib import Path
from race_collection.development_examples import freeze_population

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True);p.add_argument('--evidence-root',type=Path,required=True)
    p.add_argument('--allocation',type=Path,required=True);p.add_argument('--allocation-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    value=freeze_population(a.index,a.evidence_root,a.allocation,a.allocation_sha256,a.output)
    print(json.dumps({'status':'POPULATION_FROZEN','intended':len(value['intended']),
                      'selected':len(value['selected_race_ids']),'synthetic':False}))
