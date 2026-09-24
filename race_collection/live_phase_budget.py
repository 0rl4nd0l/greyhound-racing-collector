from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class LiveBudget:
    publication_age: float = 270.0
    refresh_seconds: float = 65.0
    work_seconds: float = 50.0
    next_trigger_seconds: float = 75.0
    overhead_seconds: float = 10.0
    handoff_overhead_seconds: float = 10.0
    handoff_poll_seconds: float = 5.0
    completion_age_seconds: float = 75.0

    @classmethod
    def for_profile(cls, profile: str | None):
        if profile is None:
            return cls()
        if profile != "bounded80-v1":
            raise ValueError("unknown_live_freshness_profile")
        return cls(refresh_seconds=80.0, completion_age_seconds=90.0)

    def age(self, observed: datetime | None, now: datetime) -> float | None:
        if observed is None:
            return None
        age = (now - observed).total_seconds()
        return age if age >= 0 else None

    def admit_work(self, observed: datetime | None, now: datetime) -> bool:
        age = self.age(observed, now)
        return (
            age is not None
            and age
            + self.work_seconds
            + self.refresh_seconds
            + self.overhead_seconds
            + self.handoff_overhead_seconds
            + self.handoff_poll_seconds
            <= self.publication_age
        )

    def safe_to_yield(self, observed: datetime | None, now: datetime) -> bool:
        age = self.age(observed, now)
        return (
            age is not None
            and age <= self.completion_age_seconds
            and age + self.next_trigger_seconds + self.refresh_seconds + self.overhead_seconds
            <= self.publication_age
        )

    def overhead_exceeded(self, elapsed: float) -> bool:
        return elapsed > self.overhead_seconds

    def overrun(self, phase: str, elapsed: float) -> bool:
        return elapsed > (self.refresh_seconds if phase == "refresh" else self.work_seconds)
