"""Study-only sealed-field dead-heat validation; production reader unchanged."""
from copy import deepcopy
from src.operator_ui.journal_results import OfficialResultSource


class ComparisonResultSource(OfficialResultSource):
    @staticmethod
    def _validate(job,bundle,race_rows,runner_rows,now):
        try:
            return OfficialResultSource._validate(job,bundle,race_rows,runner_rows,now)
        except ValueError as exc:
            if str(exc) not in {'OFFICIAL_RESULT_FINISH_AMBIGUOUS','OFFICIAL_RESULT_WINNER_MISMATCH'}:raise
        # Accept only a complete competition ranking (1,1,3,...), with explicit
        # co-winner flags and the official summary winner among those co-winners.
        positions=[r['finish_position'] for r in runner_rows]
        if any(type(p) is not int or p<1 for p in positions):raise ValueError('COMPARISON_FINISH_INVALID')
        for position in set(positions):
            if position!=1+sum(p<position for p in positions):raise ValueError('COMPARISON_FINISH_INVALID')
        winners=[r for r in runner_rows if r['finish_position']==1]
        if not winners or len(race_rows)!=1:raise ValueError('COMPARISON_FINISH_INVALID')
        if any(type(r['is_winner']) is not bool or r['is_winner']!=(r['finish_position']==1) for r in runner_rows):raise ValueError('COMPARISON_WINNER_INVALID')
        race=race_rows[0]
        if (race['winner_box'],race['winner_name']) not in {(r['box_number'],r['dog_name']) for r in winners}:raise ValueError('COMPARISON_WINNER_INVALID')
        if race['box_order']!=[r['box_number'] for r in sorted(runner_rows,key=lambda r:(r['finish_position'],race['box_order'].index(r['box_number'])))]:
            raise ValueError('COMPARISON_ORDER_INVALID')
        # Reuse ALL original field/source/time/native-ID validation with a local
        # ranking-normalized copy. Original retained rows remain untouched.
        rows=deepcopy(sorted(runner_rows,key=lambda r:(r['finish_position'],r['box_number'])))
        normalized=deepcopy(race)
        for i,row in enumerate(rows,1):row.update(finish_position=i,is_winner=i==1)
        normalized.update(winner_box=rows[0]['box_number'],winner_name=rows[0]['dog_name'],box_order=[r['box_number'] for r in rows])
        OfficialResultSource._validate(job,bundle,[normalized],rows,now)
