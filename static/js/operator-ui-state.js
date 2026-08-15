(function (root, factory) {
  const exported = factory();
  if (typeof module === "object" && module.exports) module.exports = exported;
  else root.OperatorUiState = exported;
})(typeof globalThis === "object" ? globalThis : this, function () {
  "use strict";
  const INTENT_KEY = "operatorUiPredictionIntentV1";
  const JOB_KEY = "operatorUiJobV1";
  const JOB_RE = /^job_[0-9a-f]{32}$/;
  const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
  const plain=value=>!!value&&typeof value==="object"&&!Array.isArray(value);
  const exact=(value,keys)=>plain(value)&&Object.keys(value).sort().join("|")===[...keys].sort().join("|");
  const id=value=>typeof value==="string"&&SAFE_ID.test(value);
  const hash=value=>typeof value==="string"&&/^[0-9a-f]{64}$/.test(value);
  const text=value=>typeof value==="string"&&value.length>0&&value.length<=512;
  const classifications=new Set(["AVAILABLE/FRESH","STALE","UNAVAILABLE/DATA_MISSING","INVALID/INTEGRITY_FAILED","DIVERGENT","NON_OPERATIONAL/AUTHENTICATION_REQUIRED","NON_OPERATIONAL/AUTHORIZATION_DENIED","NON_OPERATIONAL/PROVIDER_ERROR","NON_OPERATIONAL/AUDIT_UNAVAILABLE"]);
  const phases=new Set(["SUBMITTED","VALIDATED","WAITING_FOR_CLAIM","CLAIMED","ATTEMPT_STARTED","RESPONSE_RECORDED","RECEIPT_VERIFIED","CONSUMED","SCORING","PRODUCER_COMPLETED","REAP_UNCONFIRMED","PREDICTION_READY","FAILED","REJECTED","EXPIRED","TIMED_OUT","CANCELLED"]);
  const terminalPhases=new Set(["PREDICTION_READY","FAILED","REJECTED","EXPIRED","TIMED_OUT","CANCELLED"]);
  function resourceEnvelope(value,expected,detail=false){if(!plain(value)||value.schema!=="operator_ui_level_1_api_v1"||value.api_version!=="v1"||value.resource!==expected||!classifications.has(value.classification)||typeof value.stale!=="boolean"||!text(value.server_observed_at)||!plain(value.evidence))return false;const tail=Object.hasOwn(value,"data")?"data":Object.hasOwn(value,"reason")?"reason":null;if(!tail||!exact(value,["api_version","classification","evidence","resource","schema","server_observed_at","stale",tail]))return false;if(detail&&value.classification==="AVAILABLE/FRESH"&&!Object.hasOwn(value,"data"))return false;return tail!=="reason"||text(value.reason);}
  function resourceErrorEnvelope(value){return exact(value,["classification"])&&value.classification==="NON_OPERATIONAL/AUTHENTICATION_REQUIRED"||exact(value,["classification","error"])&&value.classification==="NON_OPERATIONAL/PROVIDER_ERROR"&&text(value.error);}
  function csrfEnvelope(value){return exact(value,["classification","csrf_token"])&&text(value.classification)&&text(value.csrf_token);}
  function capabilityEnvelope(value){return exact(value,["schema","authorized","runtime_configured","level"])&&value.schema==="operator_ui_r3_capability_v1"&&typeof value.authorized==="boolean"&&typeof value.runtime_configured==="boolean"&&value.level===2;}
  function errorEnvelope(value){return exact(value,["schema","classification"])&&value.schema==="operator_ui_prediction_error_v1"&&text(value.classification);}
  function timelineEvent(event){return exact(event,["event_id","phase","event_at","status","reason","event_hash","facts"])&&id(event.event_id)&&phases.has(event.phase)&&text(event.event_at)&&text(event.status)&&text(event.reason)&&hash(event.event_hash)&&plain(event.facts);}
  function verifiedResult(result){return exact(result,["schema","verification_status","probabilities","evidence"])&&result.schema==="operator_ui_verified_prediction_result_v1"&&result.verification_status==="VERIFIED"&&Array.isArray(result.probabilities)&&result.probabilities.length>0&&result.probabilities.every(row=>exact(row,["rank","runner_id","box","name","probability"])&&Number.isInteger(row.rank)&&row.rank>0&&Number.isInteger(row.box)&&row.box>0&&id(row.runner_id)&&text(row.name)&&typeof row.probability==="number"&&Number.isFinite(row.probability)&&row.probability>=0&&row.probability<=1)&&plain(result.evidence);}
  function jobEnvelope(value){if(!plain(value)||value.schema!=="operator_ui_prediction_job_response_v1"||!id(value.job_id)||!JOB_RE.test(value.job_id)||!phases.has(value.phase)||typeof value.terminal!=="boolean"||value.terminal!==terminalPhases.has(value.phase)||!id(value.race_id)||!text(value.jump_timestamp)||!hash(value.runner_set_sha256)||!id(value.model_id)||!id(value.resolved_model_identity)||!id(value.config_id)||!id(value.odds_source_id)||!Array.isArray(value.timeline)||!value.timeline.every(timelineEvent))return false;const optional=Object.hasOwn(value,"blocker")?"blocker":null;if(!exact(value,["schema","job_id","phase","terminal","race_id","jump_timestamp","runner_set_sha256","model_id","resolved_model_identity","config_id","odds_source_id","timeline","result",...(optional?[optional]:[])]))return false;if(value.phase==="PREDICTION_READY")return value.result===null?text(value.blocker):verifiedResult(value.result)&&!optional;return value.result===null&&(!value.terminal||text(value.blocker));}

  async function readAuthorityResponse(response,onLoss,validate=()=>true) {
    let lost=false;const lose=()=>{if(!lost){lost=true;onLoss();}};
    if([401,403,404].includes(response.status))lose();
    const contentType=response.headers.get("content-type")||"";
    if(!contentType.toLowerCase().includes("application/json")){lose();throw Object.assign(new Error("INVALID_RESPONSE_MEDIA"),{stable:true});}
    let payload;try{payload=await response.json();}catch(_){lose();throw Object.assign(new Error("INVALID_RESPONSE_JSON"),{stable:true});}
    if(!response.ok)lose();
    if(!validate(payload)){lose();throw Object.assign(new Error("INVALID_AUTHORITY_SCHEMA"),{stable:true});}
    return payload;
  }

  function createOperatorState(options) {
    const storage = options.storage;
    const randomUUID = options.randomUUID;
    const setTimer = options.setTimer || setTimeout;
    const clearTimer = options.clearTimer || clearTimeout;
    const getJob = options.getJob || (async () => { throw new Error("JOB_READER_UNAVAILABLE"); });
    const getCapability = options.getCapability || (async () => { throw new Error("CAPABILITY_READER_UNAVAILABLE"); });
    const onJob = options.onJob || (() => {});
    const onCapability = options.onCapability || (() => {});
    const onExhausted = options.onExhausted || (() => {});
    const onAuthorizationExhausted = options.onAuthorizationExhausted || (() => {});
    const maximum = 6;
    let capability = false;
    let timer = null;
    let inFlight = false;
    let attempts = 0;
    let recoveryTimer = null;
    let recoveryInFlight = false;
    let recoveryAttempts = 0;

    function parseIntent() {
      try {
        const value = JSON.parse(storage.getItem(INTENT_KEY) || "null");
        if (!value || value.schema !== "operator_ui_prediction_intent_v1" ||
            typeof value.idempotency_key !== "string" || !value.selection) return null;
        return value;
      } catch (_) { return null; }
    }
    function intent() { const value = parseIntent(); return value && { selection: value.selection, idempotency_key: value.idempotency_key }; }
    function jobId() { const value = storage.getItem(JOB_KEY); return JOB_RE.test(value || "") ? value : null; }
    function setCapability(envelope) {
      capability = !!envelope && envelope.authorized === true && envelope.runtime_configured === true && envelope.level === 2;
      if (!capability) stopReconnect();
      onCapability(capability);
      return capability;
    }
    function stopRecovery() { if (recoveryTimer !== null) clearTimer(recoveryTimer); recoveryTimer=null; recoveryInFlight=false; }
    function loseCapability() { capability = false; stopReconnect(); stopRecovery(); onCapability(false); }
    function canSubmit() { return capability && intent() === null && jobId() === null; }
    function beginSubmission(selection) {
      if (!capability) throw new Error("CAPABILITY_UNAVAILABLE");
      if (intent()) throw new Error("UNRESOLVED_INTENT");
      if (jobId()) throw new Error("ACTIVE_JOB");
      const value = { schema: "operator_ui_prediction_intent_v1", selection: { ...selection }, idempotency_key: randomUUID() };
      storage.setItem(INTENT_KEY, JSON.stringify(value));
      return intent();
    }
    function responseLost() {}
    function retransmission() {
      if (!capability) throw new Error("CAPABILITY_UNAVAILABLE");
      const value = intent();
      if (!value) throw new Error("NO_UNRESOLVED_INTENT");
      return value;
    }
    function associateJob(id) {
      if (!JOB_RE.test(id)) throw new Error("INVALID_JOB_ID");
      storage.setItem(JOB_KEY, id); storage.removeItem(INTENT_KEY);
    }
    function stableRejection() { storage.removeItem(INTENT_KEY); }
    function clearTerminalJob() { storage.removeItem(JOB_KEY); stopReconnect(); }
    function stopReconnect() { if (timer !== null) clearTimer(timer); timer = null; inFlight = false; }
    function recoverAuthority(id) {
      if (recoveryTimer!==null||recoveryInFlight||recoveryAttempts>=3||!JOB_RE.test(id)) return;
      recoveryTimer=setTimer(async()=>{
        recoveryTimer=null;if(recoveryInFlight)return;recoveryInFlight=true;
        try {
          const envelope=await getCapability();
          if (!setCapability(envelope)) throw new Error("CAPABILITY_DENIED");
          recoveryAttempts=0;attempts=0;recoveryInFlight=false;reconnect(id);
        } catch (_) {
          recoveryAttempts+=1;recoveryInFlight=false;
          if(recoveryAttempts<3)recoverAuthority(id);else onAuthorizationExhausted();
        }
      },Math.min(12000,1500*(2**recoveryAttempts)));
    }
    function exhaust(id) { capability=false;stopReconnect();onCapability(false);onExhausted();recoverAuthority(id); }
    function schedule(id, delay) {
      if (!capability || timer !== null || inFlight || attempts >= maximum || !JOB_RE.test(id)) return;
      timer = setTimer(async () => {
        timer = null;
        if (!capability || inFlight) return;
        inFlight = true;
        try {
          const payload = await getJob(id);
          if (!payload || payload.job_id !== id) throw Object.assign(new Error("JOB_IDENTITY_MISMATCH"), { stable: true });
          attempts = 0; onJob(payload);
          if (payload.terminal) clearTerminalJob(); else { inFlight = false; schedule(id, 1500); }
        } catch (error) {
          attempts += 1;
          if (error && error.stable) loseCapability();
          else if(attempts>=maximum) { inFlight=false; exhaust(id); }
          else { inFlight = false; schedule(id, Math.min(12000, 750 * (2 ** attempts))); }
        } finally { inFlight = false; }
      }, delay);
    }
    function reconnect(id = jobId()) { schedule(id, 0); }
    return { intent, jobId, setCapability, loseCapability, canSubmit, beginSubmission,
      responseLost, retransmission, associateJob, stableRejection, clearTerminalJob,
      reconnect, stopReconnect, transportAttempts: () => attempts };
  }
  return { createOperatorState, readAuthorityResponse, resourceEnvelope, csrfEnvelope,
    capabilityEnvelope, errorEnvelope, resourceErrorEnvelope, timelineEvent, verifiedResult, jobEnvelope,
    INTENT_KEY, JOB_KEY };
});
