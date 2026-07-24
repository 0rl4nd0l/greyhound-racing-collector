ALTER TABLE sealed_evidence ADD COLUMN request_intent_digest TEXT
    CHECK (request_intent_digest IS NULL OR (
        length(request_intent_digest) = 71 AND
        substr(request_intent_digest, 1, 7) = 'sha256:' AND
        substr(request_intent_digest, 8) NOT GLOB '*[^0-9a-f]*'
    ));

CREATE TRIGGER sealed_evidence_requires_request_intent_insert
BEFORE INSERT ON sealed_evidence
WHEN NEW.request_intent_digest IS NULL
BEGIN SELECT RAISE(ABORT, 'sealed_evidence request intent is required'); END;
