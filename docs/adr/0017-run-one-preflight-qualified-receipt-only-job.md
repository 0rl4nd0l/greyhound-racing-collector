# Run one preflight-qualified receipt-only job

After the accepted R3 runtime is healthy, one acceptance job may be submitted only when a fresh verified current-index race agrees with a fresh sealed receipt. It uses `latest-research` resolving to `market_form_residual_v1`, the checked-in `manual-default` configuration, and `odds_source=receipt`. If the job blocks or fails, execution stops and reports its evidence; there is no automatic retry, capture, fallback model, EV calculation, or betting action.
