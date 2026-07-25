# Use one authoritative race collection scheduler

One Race Collection Service will own the durable racing-day state machine and internally schedule race-card collection, adaptive pre-jump odds capture, evidence sealing, deferred prediction batches, and subsequent result collection. The process supervisor may start and restart this service, but overlapping timers and legacy automation may not independently perform those operations; manual commands remain adapters to the same state machine.
