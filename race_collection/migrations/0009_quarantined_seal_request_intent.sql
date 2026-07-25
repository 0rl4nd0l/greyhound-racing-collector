ALTER TABLE collection_quarantines ADD COLUMN request_intent_digest TEXT
    CHECK (request_intent_digest IS NULL OR (
        length(request_intent_digest) = 71 AND
        substr(request_intent_digest, 1, 7) = 'sha256:' AND
        substr(request_intent_digest, 8) NOT GLOB '*[^0-9a-f]*'
    ));

CREATE TRIGGER collection_quarantines_requires_sealing_request_intent_insert
BEFORE INSERT ON collection_quarantines
WHEN (NEW.stage = 'sealing' AND NEW.request_intent_digest IS NULL)
  OR (NEW.stage <> 'sealing' AND NEW.request_intent_digest IS NOT NULL)
BEGIN SELECT RAISE(ABORT, 'collection quarantine request intent disagrees with stage'); END;
