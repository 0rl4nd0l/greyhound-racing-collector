CREATE TABLE phase7_alert_resolutions (
 alert_id TEXT PRIMARY KEY REFERENCES phase7_alerts(alert_id),
 resolved_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);
CREATE TRIGGER phase7_alert_resolution_exact BEFORE INSERT ON phase7_alert_resolutions WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_resolve_alert')
BEGIN SELECT RAISE(ABORT,'alert resolution lacks administrative command authority'); END;
CREATE TRIGGER phase7_alert_resolved_at_exact BEFORE UPDATE OF resolved_at ON phase7_alerts WHEN
 NEW.resolved_at IS NOT NULL AND NOT EXISTS (SELECT 1 FROM phase7_alert_resolutions r
  WHERE r.alert_id=NEW.alert_id AND r.resolved_at=NEW.resolved_at)
BEGIN SELECT RAISE(ABORT,'alert cannot be resolved by direct SQL'); END;
CREATE TRIGGER phase7_alert_resolutions_append_only_update BEFORE UPDATE ON phase7_alert_resolutions
BEGIN SELECT RAISE(ABORT,'alert resolutions are append-only'); END;
CREATE TRIGGER phase7_alert_resolutions_append_only_delete BEFORE DELETE ON phase7_alert_resolutions
BEGIN SELECT RAISE(ABORT,'alert resolutions are append-only'); END;

CREATE TRIGGER phase7_admin_audit_exact BEFORE INSERT ON phase7_admin_audit WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id AND (
  (NEW.command IN ('pause','resume') AND o.kind='phase7_admin_pause') OR
  (NEW.command='resolve_alert' AND o.kind='phase7_resolve_alert') OR
  (NEW.command='reset' AND o.kind='phase7_reset_probation') OR
  (NEW.command='initialize_legacy' AND o.kind='phase7_initialize_legacy') OR
  (NEW.command='activate' AND o.kind='phase7_activate_release') OR
  (NEW.command='rollback' AND o.kind='phase7_rollback_release')))
BEGIN SELECT RAISE(ABORT,'admin audit lacks exact application command authority'); END;

CREATE TRIGGER phase7_pause_results BEFORE INSERT ON result_attempts WHEN
 EXISTS (SELECT 1 FROM phase7_pauses WHERE scope='results' AND paused=1)
BEGIN SELECT RAISE(ABORT,'results are administratively paused'); END;
CREATE TRIGGER phase7_pause_joins BEFORE INSERT ON training_examples WHEN
 EXISTS (SELECT 1 FROM phase7_pauses WHERE scope='joins' AND paused=1)
BEGIN SELECT RAISE(ABORT,'training joins are administratively paused'); END;
CREATE TRIGGER phase7_pause_training_requests BEFORE INSERT ON phase6_training_requests WHEN
 EXISTS (SELECT 1 FROM phase7_pauses WHERE scope='training_requests' AND paused=1)
BEGIN SELECT RAISE(ABORT,'training requests are administratively paused'); END;
CREATE TRIGGER phase7_pause_service_training_requests BEFORE INSERT ON phase6_service_training_requests WHEN
 EXISTS (SELECT 1 FROM phase7_pauses WHERE scope='training_requests' AND paused=1)
BEGIN SELECT RAISE(ABORT,'training requests are administratively paused'); END;
CREATE TRIGGER phase7_pause_promotion BEFORE INSERT ON phase6_promotion_records WHEN
 EXISTS (SELECT 1 FROM phase7_pauses WHERE scope='promotion' AND paused=1)
BEGIN SELECT RAISE(ABORT,'promotion is administratively paused'); END;
