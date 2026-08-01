"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { createOperatorState, readAuthorityResponse } = require("../../static/js/operator-ui-state.js");

function storage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem: key => values.has(key) ? values.get(key) : null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: key => values.delete(key),
  };
}

const A = { race_id: "race-a", model_id: "model-a", config_id: "config-a", odds_source_id: "auto" };
const B = { ...A, race_id: "race-b" };

test("lost response freezes selection and retransmits exactly the persisted key", () => {
  const store = storage();
  const state = createOperatorState({ storage: store, randomUUID: () => "key-1" });
  state.setCapability({ authorized: true, runtime_configured: true, level: 2 });
  assert.deepEqual(state.beginSubmission(A), { selection: A, idempotency_key: "key-1" });
  state.responseLost();
  assert.equal(state.canSubmit(B), false);
  assert.throws(() => state.beginSubmission(B), /UNRESOLVED_INTENT/);
  assert.deepEqual(state.retransmission(), { selection: A, idempotency_key: "key-1" });

  const refreshed = createOperatorState({ storage: store, randomUUID: () => "key-2" });
  refreshed.setCapability({ authorized: true, runtime_configured: true, level: 2 });
  assert.equal(refreshed.canSubmit(B), false);
  assert.deepEqual(refreshed.retransmission(), { selection: A, idempotency_key: "key-1" });
});

test("job association and stable rejection are the only intent release points", () => {
  const store = storage();
  const state = createOperatorState({ storage: store, randomUUID: () => "key-1" });
  state.setCapability({ authorized: true, runtime_configured: true, level: 2 });
  state.beginSubmission(A);
  state.associateJob("job_0123456789abcdef0123456789abcdef");
  assert.equal(state.intent(), null);
  assert.equal(state.jobId(), "job_0123456789abcdef0123456789abcdef");

  state.clearTerminalJob();
  state.beginSubmission(B);
  state.stableRejection("SELECTION_NOT_ALLOWLISTED");
  assert.equal(state.intent(), null);
  assert.equal(state.canSubmit(A), true);
});

test("reconnect owns one non-overlapping timer and never posts", async () => {
  const timers = [];
  const requests = [];
  const state = createOperatorState({
    storage: storage({ operatorUiJobV1: "job_0123456789abcdef0123456789abcdef" }),
    randomUUID: () => "unused",
    setTimer: (fn, delay) => (timers.push({ fn, delay }), timers.length),
    clearTimer: () => {},
    getJob: async id => (requests.push(id), { job_id: id, terminal: false }),
  });
  state.setCapability({ authorized: true, runtime_configured: true, level: 2 });
  state.reconnect();
  state.reconnect();
  assert.equal(timers.length, 1);
  await timers.shift().fn();
  assert.equal(requests.length, 1);
  assert.equal(timers.length, 1);
  assert.equal(state.transportAttempts(), 0);
});

test("capability loss cancels reconnect and blocks every submission", () => {
  let cancelled = 0;
  const state = createOperatorState({
    storage: storage(), randomUUID: () => "key-1",
    setTimer: () => 7, clearTimer: () => { cancelled += 1; },
  });
  state.setCapability({ authorized: true, runtime_configured: true, level: 2 });
  assert.equal(state.canSubmit(A), true);
  state.reconnect("job_0123456789abcdef0123456789abcdef");
  state.loseCapability("SESSION_EXPIRED");
  assert.equal(cancelled, 1);
  assert.equal(state.canSubmit(A), false);
  assert.throws(() => state.beginSubmission(A), /CAPABILITY_UNAVAILABLE/);
});

test("transport backoff is bounded and exhausts after six exact-job reads", async () => {
  const timers = [], delays = [], reads = [];
  const job = "job_0123456789abcdef0123456789abcdef";
  const state = createOperatorState({ storage: storage({ operatorUiJobV1: job }), randomUUID: () => "unused",
    setTimer: (fn, delay) => (timers.push(fn), delays.push(delay), timers.length), clearTimer: () => {},
    getJob: async id => { reads.push(id); throw new Error("offline"); } });
  state.setCapability({ authorized: true, runtime_configured: true, level: 2 }); state.reconnect();
  while (timers.length) await timers.shift()();
  assert.deepEqual(delays, [0, 1500, 3000, 6000, 12000, 12000, 1500, 3000, 6000]);
  assert.deepEqual(reads, Array(6).fill(job));
  assert.equal(state.transportAttempts(), 6);
});

test("successful recovery resets only transport backoff and terminal cleanup removes the exact job", async () => {
  const timers = []; let reads = 0;
  const job = "job_0123456789abcdef0123456789abcdef"; const store = storage({ operatorUiJobV1: job });
  const state = createOperatorState({ storage: store, randomUUID: () => "unused",
    setTimer: fn => (timers.push(fn), timers.length), clearTimer: () => {},
    getJob: async id => { assert.equal(id, job); reads += 1; if (reads === 1) throw new Error("offline"); return { job_id: id, terminal: reads === 3 }; } });
  state.setCapability({ authorized: true, runtime_configured: true, level: 2 }); state.reconnect();
  await timers.shift()(); await timers.shift()();
  assert.equal(state.transportAttempts(), 0); assert.equal(state.jobId(), job);
  await timers.shift()();
  assert.equal(state.jobId(), null); assert.equal(timers.length, 0);
});

test("stable auth or not-found response stops timers and disables without substituting a job", async () => {
  for (const code of ["AUTHENTICATION_REQUIRED", "JOB_NOT_FOUND"]) {
    const timers = []; const job = "job_0123456789abcdef0123456789abcdef";
    const state = createOperatorState({ storage: storage({ operatorUiJobV1: job }), randomUUID: () => "unused",
      setTimer: fn => (timers.push(fn), timers.length), clearTimer: () => {},
      getJob: async id => { assert.equal(id, job); throw Object.assign(new Error(code), { stable: true }); } });
    state.setCapability({ authorized: true, runtime_configured: true, level: 2 }); state.reconnect();
    await timers.shift()();
    assert.equal(timers.length, 0); assert.equal(state.canSubmit(A), false); assert.equal(state.jobId(), job);
  }
});

test("exhaustion fails closed and bounded GET-only capability recovery resumes the same job", async () => {
  const timers=[]; const job="job_0123456789abcdef0123456789abcdef"; let exhausted=0, refreshes=0, reads=0;
  const state=createOperatorState({storage:storage({operatorUiJobV1:job}),randomUUID:()=>"unused",
    setTimer:(fn,delay)=>(timers.push({fn,delay}),timers.length),clearTimer:()=>{},
    getJob:async id=>{reads+=1;if(reads<=6)throw new Error("offline");return{job_id:id,terminal:false};},
    getCapability:async()=>{refreshes+=1;return{authorized:true,runtime_configured:true,level:2};},
    onExhausted:()=>{exhausted+=1;}});
  state.setCapability({authorized:true,runtime_configured:true,level:2});state.reconnect();
  for(let i=0;i<6;i++)await timers.shift().fn();
  assert.equal(exhausted,1);assert.equal(state.canSubmit(A),false);assert.equal(state.jobId(),job);
  while(timers.length&&refreshes===0)await timers.shift().fn();
  assert.equal(refreshes,1);assert.equal(state.jobId(),job);
  await timers.shift().fn();
  assert.equal(reads,7);assert.equal(state.transportAttempts(),0);
});

test("six transport and three authorization failures preserve the same job and finish accessibly", async () => {
  const timers=[]; const job="job_0123456789abcdef0123456789abcdef"; let reads=0, capabilityReads=0, final=0, posts=0;
  const state=createOperatorState({storage:storage({operatorUiJobV1:job}),randomUUID:()=>{posts+=1;return "unused";},
    setTimer:(fn,delay)=>(timers.push({fn,delay}),timers.length),clearTimer:()=>{},
    getJob:async()=>{reads+=1;throw new Error("offline");},getCapability:async()=>{capabilityReads+=1;throw new Error("denied");},
    onAuthorizationExhausted:()=>{final+=1;}});
  state.setCapability({authorized:true,runtime_configured:true,level:2});state.reconnect();
  while(timers.length||reads<6||capabilityReads<3){const timer=timers.shift();assert.ok(timer);await timer.fn();}
  assert.equal(reads,6);assert.equal(capabilityReads,3);assert.equal(final,1);assert.equal(state.jobId(),job);assert.equal(posts,0);assert.equal(timers.length,0);
});

test("every authority response failure class loses capability before disclosure", async()=>{
  const cases=[
    {status:401,ok:false,type:"application/json",json:async()=>({classification:"AUTH"})},
    {status:403,ok:false,type:"application/json",json:async()=>({classification:"DENIED"})},
    {status:404,ok:false,type:"application/json",json:async()=>({classification:"MISSING"})},
    {status:500,ok:false,type:"application/json",json:async()=>({classification:"FAILED"})},
    {status:200,ok:true,type:"text/html",json:async()=>({})},
    {status:200,ok:true,type:"application/json",json:async()=>{throw new Error("malformed");}},
    {status:200,ok:true,type:"application/json",json:async()=>({schema:"wrong"}),invalid:true},
  ];
  for(const item of cases){let losses=0;const response={status:item.status,ok:item.ok,headers:{get:()=>item.type},json:item.json};
    if(item.ok&&!item.invalid&&item.type.includes("json"))continue;
    await assert.rejects(()=>readAuthorityResponse(response,()=>{losses+=1},value=>value.schema==="expected"));assert.equal(losses,1);
  }
});
