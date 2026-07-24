DROP TRIGGER phase7_rejected_result_exact;
CREATE TRIGGER phase7_rejected_result_exact BEFORE INSERT ON phase7_rejected_result_commands WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_reject_result_before_prediction')
 OR NOT EXISTS (SELECT 1 FROM races r WHERE r.race_id=NEW.race_id
  AND r.racing_day_id=NEW.racing_day_id)
 OR EXISTS (SELECT 1 FROM deferred_predictions p WHERE p.race_id=NEW.race_id)
 OR NOT EXISTS (SELECT 1 FROM phase7_alerts a
  WHERE a.alert_id='result-rejection:'||NEW.operation_id
   AND a.operation_id=NEW.operation_id AND a.racing_day_id=NEW.racing_day_id
   AND a.category='result_before_prediction' AND a.resolved_at IS NULL)
 OR NOT EXISTS (SELECT 1 FROM phase7_pauses p WHERE p.scope='results' AND p.paused=1
  AND p.operation_id=NEW.operation_id AND p.reason='alert:result_before_prediction')
BEGIN SELECT RAISE(ABORT,'result rejection lacks its atomic alert and scoped pause'); END;

CREATE TRIGGER phase7_result_rejection_alert_exact BEFORE INSERT ON phase7_alerts WHEN
 NEW.category='result_before_prediction' AND NEW.alert_id LIKE 'result-rejection:%' AND
 (NEW.alert_id<>'result-rejection:'||NEW.operation_id OR
  NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
   AND o.kind='phase7_reject_result_before_prediction'))
BEGIN SELECT RAISE(ABORT,'result rejection alert lacks exact application authority'); END;
