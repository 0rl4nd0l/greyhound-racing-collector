ALTER TABLE phase7_backups ADD COLUMN artifact_reference_contract TEXT NOT NULL DEFAULT 'phase7-artifact-references-v1'
 CHECK(artifact_reference_contract='phase7-artifact-references-v1');
