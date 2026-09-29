from datetime import date, datetime, timezone, timedelta
import pytest
from scripts.audit_retained_history_depth import known_context, merge_diagnostics, safe_path
from scripts.reconstruct_asof_card_history import union_history, eligible_prior_history
from scripts.run_shadow_non_tgr_rf_evaluation import merge_prior_history_rows


def start(day=1,finish=2,distance=450,**extra):
    return {'race_date':f'2026-01-{day:02d}','venue':'V','distance_num':distance,
            'grade':'5','grade_normalized':'5','finish_num':finish,'time_num':26,**extra}


def test_depth_and_formula_changes_are_separate():
    rows=[start(finish=None,distance=400),start(day=2,finish=1,distance=450)]
    context=known_context(rows,'V',450,'5')
    assert context['all']['starts']==2
    assert context['all']['known_finishes']==1
    assert context['distance_exact']['starts']==1
    assert context['distance_tolerance']['starts']==2
    assert context['distance_tolerance']['known_finishes']==1


def test_duplicate_precedence_and_unreconciled_conflict_are_visible():
    db=[start()];card=[start(),start(finish=3)]
    merged,diagnostic=merge_diagnostics(db,card,merge_prior_history_rows)
    assert merged[0] is db[0]
    assert diagnostic['duplicate_rows_removed']==1
    assert diagnostic['possible_same_start_conflict_groups']==1
    assert diagnostic['retained_db_rows']==1


def test_read_path_cannot_escape_bundle(tmp_path):
    with pytest.raises(ValueError,match='outside'):
        safe_path(tmp_path,'../protected.db')


def h(day,finish=2):
    return {'date':date(2026,1,day),'venue':'V','distance':450,'grade':'5','finish':finish,'margin':1}


def test_reconstructed_history_dedup_and_conflicts():
    assert union_history([h(2)],[h(2),h(1)]) == [h(2),h(1)]
    assert union_history([h(2)],[h(2,finish=3)]) is None


def test_reconstruction_keeps_original_twenty_start_cap():
    assert len(union_history([h(30)],[h(d) for d in range(1,30)]))==20
    assert union_history([h(30)],[h(d) for d in range(1,30)])[-1]['date']==date(2026,1,11)


def test_later_and_equal_capture_are_never_borrowed():
    cutoff=datetime(2026,1,10,tzinfo=timezone.utc)
    pool=[(cutoff-timedelta(seconds=1),'before',[h(1),h(10)]),
          (cutoff,'equal',[h(2)]),(cutoff+timedelta(seconds=1),'after',[h(3)])]
    admitted,history=eligible_prior_history(pool,cutoff,date(2026,1,10))
    assert [rid for _,rid,_ in admitted]==['before']
    assert history==[h(1)]
