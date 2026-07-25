DROP TRIGGER phase7_cutover_exact;
CREATE TRIGGER phase7_cutover_exact BEFORE INSERT ON phase7_cutover_eligibility WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_cutover_eligibility')
 OR NOT EXISTS (SELECT 1 FROM phase7_operational_evidence a
  WHERE a.artifact_checksum=NEW.evidence_checksum AND a.evidence_kind='cutover')
 OR NOT EXISTS (
  SELECT 1 FROM phase7_day_evidence first
  JOIN phase7_day_evidence second ON second.racing_day_id=NEW.second_racing_day_id
  JOIN phase6_racing_day_schedule schedule ON schedule.racing_day_id=second.racing_day_id
  WHERE first.racing_day_id=NEW.first_racing_day_id
   AND first.complete=1 AND first.critical_failure=0
   AND second.complete=1 AND second.critical_failure=0
   AND first.release_id=NEW.candidate_release_id
   AND second.release_id=NEW.candidate_release_id
   AND schedule.predecessor_racing_day_id=first.racing_day_id)
BEGIN SELECT RAISE(ABORT,'cutover lacks two consecutive authenticated scheduled Racing Days'); END;
