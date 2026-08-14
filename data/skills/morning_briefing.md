---
name: morning_briefing
description: Summarize daily priorities and habits without exposing financial or deep archival logs.
allowed-tools:
  - get_core_fact
allowed-lockers:
  - user_profile
  - daily_goals
---

# Morning Briefing Protocol
1. Read the `daily_goals` locker using `get_core_fact`.
2. Greet the user concisely and state the top 3 goals for today.
3. Ask if they want to adjust any priorities before starting work.