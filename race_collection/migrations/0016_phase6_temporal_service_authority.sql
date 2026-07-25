DROP TRIGGER phase6_forecast_service_before_result;
CREATE TRIGGER phase6_forecast_service_before_result BEFORE INSERT ON phase6_forecast_service_artifacts
WHEN NEW.deferred_prediction_id IS NULL OR NOT EXISTS (
 SELECT 1 FROM phase6_runs s
 JOIN deferred_predictions p ON p.prediction_id=NEW.deferred_prediction_id
 JOIN expected_races e ON e.race_id=NEW.race_id
 JOIN races r ON r.race_id=p.race_id
 JOIN racing_days d ON d.racing_day_id=r.racing_day_id
 JOIN sealed_evidence z ON z.seal_id=p.seal_id AND z.race_id=p.race_id
 WHERE s.run_id=NEW.service_run_id AND s.run_kind='forecast_service'
   AND p.race_id=NEW.race_id AND p.evidence_checksum=NEW.evidence_checksum
   AND p.computed_at=NEW.generated_at AND s.started_at=NEW.generated_at
   AND d.closed_at IS NOT NULL AND d.closed_at<=NEW.generated_at
   AND e.programme_checksum IS NOT NULL
   AND z.normalized_checksum=NEW.evidence_checksum
   AND NOT EXISTS (SELECT 1 FROM result_attempts x WHERE x.race_id=NEW.race_id AND x.attempted_at<=NEW.generated_at)
)
BEGIN SELECT RAISE(ABORT,'forecast lacks its genuine pre-result Phase-3 prediction authority'); END;
