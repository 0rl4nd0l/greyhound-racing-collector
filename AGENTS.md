# Project guidance

Use the globally installed Matt Pocock skills for planning, implementation,
research, testing, and review.

Preserve existing tracked and untracked work. Use ordinary Git inspection and
focused validation. Ask before external writes, runtime or service changes,
data, model, or betting changes, or destructive operations.

## Website access

Before accessing a website, check the retained evidence for that site's working
access method. Reuse the verified route and session setup, including the browser
profile, authentication and headed mode when required. Prefer the established
browser workflow over a fresh plain-HTTP or headless probe. Use direct HTTP/API
access when that method is verified for the specific source and task.

When no working method is established, state that gap and inspect the existing
browser workflow before choosing a new probe. Record the method and scope of
any failure: one HTTP 403 does not establish that every access path is blocked.
Preserve source holds and request limits; a method change must not bypass a
denial, challenge or authorization boundary. Never expose session secrets.
