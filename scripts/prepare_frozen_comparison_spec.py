"""Prepare a non-loadable proposal using only old, already-inspected variance."""
from pathlib import Path
import json,hashlib,csv,math,statistics
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]

def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    paths=['docs/research/offline_20260924_access_incident.json','docs/research/offline_systematic_evidence/protected_records.json','docs/forward_overround_successor_protocol.md','docs/sportsbet_betfair_forward_consensus_protocol.md','docs/agent_tasks/prospective_market_form_residual_cohort_v2_20260716.md','docs/prospective_residual_evaluation_assessment.md']
    external=Path('/home/l4nd0/greyhound-prospective-readiness-20260916/PROSPECTIVE_EVALUATION_PROPOSAL.md')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    review={'schema_version':'four_way_reservation_review_v1','status':'PREPARED_NO_ALLOCATION','sources':{p:sha(ROOT/p) for p in paths},'separate_residual_proposal':{'path':str(external),'sha256':sha(external)},'findings':[
    'July protected/closed study membership remains excluded; no recovery or outcome access.',
    'Sportsbet/Betfair August 18 predecessor and August 20-September 30 replacement remain protected.',
    'October overround successor has unbounded first-1000 accrual after separately approved activation, earliest October 1; November is NOT automatically available.',
    'September 16 separate residual proposal is PROPOSED_NOT_AUTHORIZED, no allocated population; must remain inactive for the proposed exclusive four-way window.',
    'Current operational acceptance and collector campaigns grant no scientific population or result access.',
    'Propose exclusive November 1 2026-April 18 2027 allocation, conditional on owner explicitly keeping both future proposals inactive until after this window or proving completed disjoint membership.',
    'No authority or activation is created by this review. Refresh all reservation sources and resolve additions before allocation.'],
    'incident':'58 reserved races / 393 labels decoded earlier; timing aggregates influenced original T-2 context. Exclusion did not reverse access. No new protected outcomes read.',
    'proposed_history_firewall':{'deny_dates_inclusive':[['2026-07-15','2026-10-31']],'effect':'conservative rejection, never filter history to rescue a prediction','required_future_authority':'machine-only strictly earlier history in exclusively allocated population; no human target-result access or interim metrics'}}
    def put(name,value):
        with (out/name).open('xb') as f:f.write((json.dumps(value,sort_keys=True,indent=2)+'\n').encode())
    put('reservation_review.json',review)
    races=defaultdict(list)
    source=ROOT/'docs/research/offline_systematic_evidence/outer_predictions.csv'
    for row in csv.DictReader(source.open()):races[row['race_id']].append(row)
    planning={}
    for name in ['p_refit_box','p_refit_half']:
        values=[];dates=defaultdict(list)
        for rr in races.values():
            w=next(r for r in rr if r['y']=='1');diff=-math.log(float(w[name]))+math.log(float(w['p_market']))
            values.append(diff);dates[w['race_date']].append(diff)
        mean=statistics.mean(values);n=len(values);d=len(dates)
        se=math.sqrt(d/(d-1)*sum(sum(x-mean for x in xs)**2 for xs in dates.values())/n**2)
        z=statistics.NormalDist().inv_cdf(1-.05/(2*4))
        planning[name]={'prior_races':n,'prior_dates':d,'paired_mean_nats':mean,'date_cluster_standard_error':se,'variance_inflation':2,'family_contrasts':4,'normal_quantile':z,'target_halfwidth_nats':.01,'estimated_required_scoreable_dates':math.ceil(d*(z*se/.01)**2*2)}
    put('precision_planning.json',{'source_sha256':sha(source),'prior_evaluation_is_development':True,'candidates':planning,'fixed_calendar_days':168,'target_scoreable_dates':119,'illustrative_races_at_prior_density':math.ceil(119*177/15),'limitations':['Only 15 historical dates; volatile variance and cross-date dependence.','Planning observed candidate-minus-market differences; candidate-minus-production precision unmeasured.','No promised power; future coverage and effect size may differ.','No interim resizing.']})
    registry=ROOT/'artifacts/research_comparison/frozen_20260924/registry.json'
    plan={'schema_version':'frozen_four_way_comparison_plan_v1','status':'PREPARED_NOT_AUTHORIZED','authority_reference':None,'exclusive_population_allocation_reference':None,'machine_history_authority_reference':None,
    'activated_at':'2026-10-31T12:00:00+11:00','starts_at':'2026-11-01T00:00:00+11:00','ends_at':'2027-04-18T00:00:00+10:00',
    'programme_root':'/home/l4nd0/greyhound-frozen-comparison-20261101/comparison','prediction_output_roots':['/home/l4nd0/greyhound-frozen-comparison-20261101/operational-predictions/bundles'],
    'decision_seconds_before_jump':120,'quote_lead_seconds':[120,600],'denied_history_intervals':[['2026-07-15','2026-10-31']],
    'candidate_registry':{'path':str(registry),'sha256':sha(registry)},'reservation_review_sha256':sha(out/'reservation_review.json'),
    'fixed_closure_days':14,'target_scoreable_dates':119,'precision_target_nats':.01,'bootstrap_seed':20260924,'bootstrap_replicates':20000,'family_contrasts':4,
    'no_interim_metrics':True,'production_promotion':False,'betting':False}
    put('prepared_plan.json',plan)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.out)
