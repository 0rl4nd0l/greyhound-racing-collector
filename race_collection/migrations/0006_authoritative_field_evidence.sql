ALTER TABLE field_evidence ADD COLUMN source TEXT NOT NULL DEFAULT 'legacy_unknown';

CREATE TRIGGER field_evidence_append_only_update
BEFORE UPDATE ON field_evidence
BEGIN SELECT RAISE(ABORT, 'field_evidence is append-only'); END;

CREATE TRIGGER field_evidence_append_only_delete
BEFORE DELETE ON field_evidence
BEGIN SELECT RAISE(ABORT, 'field_evidence is append-only'); END;
