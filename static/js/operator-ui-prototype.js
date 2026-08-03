(() => {
  "use strict";

  const fixtures = {
    valid: ["Exact identity and ordered active runner set are verified.", false, "ok"],
    ambiguous: ["Disabled — ambiguous race identity; choose one exact source-bound race.", true, "error"],
    "post-jump": ["Disabled — scheduled jump has passed; prediction evidence must be strictly pre-jump.", true, "error"],
    "missing-runner": ["Disabled — an active runner identity is missing.", true, "error"],
    stale: ["Disabled — source evidence exceeds the 300-second pre-jump freshness policy.", true, "error"],
    "missing-jump": ["Disabled — scheduled jump identity is missing.", true, "error"],
    unsupported: ["Disabled — capture window or model configuration is unsupported.", true, "error"],
    conflicting: ["Disabled — source and selected-race evidence conflict.", true, "error"],
    unavailable: ["Disabled — required source evidence is unavailable.", true, "error"],
  };
  const state = document.querySelector("#fixture-state");
  const review = document.querySelector("#review-selection");
  const reason = document.querySelector("#selection-reason");
  const dialog = document.querySelector("#confirmation-dialog");
  const confirm = document.querySelector("#confirm-fixture");
  const printMedia = window.matchMedia("print");
  let detailsOpenStates = null;

  function enterPrintMode() {
    if (detailsOpenStates !== null) return;
    detailsOpenStates = new Map();
    document.querySelectorAll("details").forEach((details) => {
      detailsOpenStates.set(details, details.open);
      details.open = true;
    });
  }

  function leavePrintMode() {
    if (detailsOpenStates === null) return;
    detailsOpenStates.forEach((wasOpen, details) => {
      details.open = wasOpen;
    });
    detailsOpenStates = null;
  }

  function projectPrintMode() {
    if (printMedia.matches) {
      enterPrintMode();
    } else {
      leavePrintMode();
    }
  }

  function projectFixture() {
    const [message, disabled, tone] = fixtures[state.value];
    reason.firstChild.textContent = `${message} `;
    reason.className = `callout callout--${tone}`;
    review.disabled = disabled;
    review.setAttribute("aria-disabled", String(disabled));
  }

  state.addEventListener("change", projectFixture);
  review.addEventListener("click", () => dialog.showModal());
  document.querySelector("[data-dialog-close]").addEventListener("click", () => dialog.close());
  confirm.addEventListener("click", () => {
    dialog.close();
    document.querySelector("#prediction-lifecycle").scrollIntoView();
    document.querySelector("#prediction-lifecycle h2").focus({ preventScroll: true });
  });
  printMedia.addEventListener("change", projectPrintMode);
  window.addEventListener("beforeprint", enterPrintMode);
  window.addEventListener("afterprint", projectPrintMode);
  projectPrintMode();
  projectFixture();
})();
