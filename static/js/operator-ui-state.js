(function (root, factory) {
  const exported = factory();
  if (typeof module === "object" && module.exports) module.exports = exported;
  else root.OperatorUiState = exported;
})(typeof globalThis === "object" ? globalThis : this, function () {
  "use strict";
  const INTENT_KEY = "operatorUiPredictionIntentV1";
  const JOB_KEY = "operatorUiJobV1";
  const JOB_RE = /^job_[0-9a-f]{32}$/;

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
          if(recoveryAttempts<3)recoverAuthority(id);
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
  return { createOperatorState, readAuthorityResponse, INTENT_KEY, JOB_KEY };
});
