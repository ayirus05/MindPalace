---
name: palace-index
description: Always-loaded map of the MindPalace — who this person is and where each piece of their dossier lives. Claude reads this first to orient every query.
type: index
updated: 2026-07-31
---

# MindPalace Index

> This file is always loaded into Claude's context. Keep it short — one screen. It orients every query; the detailed records live in the files below and are pulled on demand.

## At a glance

- **Name:** _(fill from `personality/profile.md`)_
- **Life stage & roles:** _()_
- **Core values:** _()_
- **Communication style:** _()_  — see `personality/communication.md`
- **What motivates them:** _()_  — see `personality/motivation.md`
- **Optimal learning style:** _()_  — see `personality/cognition.md`

## Palace map

### Personality tier (populated via the questionnaire app)
Base profile and the eight personality domains. Read the relevant domain file(s) before answering a query in that area.

| Domain | File | What it captures |
|---|---|---|
| Identity & roles | `personality/profile.md` | Demographics, life stage, roles, self-narrative |
| Personality traits | `personality/personality.md` | Big Five markers, temperament, dispositions |
| Values & philosophy | `personality/values.md` | Core values, beliefs, money/health mindset |
| Communication style | `personality/communication.md` | Tone, formality, directness, information preferences |
| Motivation & drive | `personality/motivation.md` | What energizes them, goals, what kills motivation |
| Cognitive patterns | `personality/cognition.md` | Decision-making, problem-solving, learning style |
| Relationships & social | `personality/relationships.md` | Social energy, circle, attachment style, conflict style |
| Stress & coping | `personality/stress-response.md` | Stress signals, coping strategies, recovery patterns |
| Life context | `personality/life-context.md` | Daily routine, energy patterns, current season of life |
| Goals & aspirations | `personality/goals.md` | Short/long-term goals, what success looks like |

### Future tiers (not yet built)
- **Structured core** — health, finance, mind domains (Tier 1 expansion)
- **Unstructured index** — journals, logs, transcripts (Tier 2, hybrid embedding layer)

## How to use this palace

1. **Read this index first** for orientation — it's always in your context.
2. **Before answering a query,** read the most relevant domain file(s) from the map above. Don't guess at the user's preferences — pull the record.
3. **After learning something new about the user,** offer to write it to the appropriate record so the palace stays current.
4. **Check the `updated:` date** on any record you read — if it's stale, flag it before relying on it.
5. **Produce synthesis, not regurgitation** — the data is an input to better answers, not the answer itself.

## Skill layer
Workflow skills live in `skills/`. Each skill tells Claude which files to read and how to synthesize for a given query type. (No skills written yet — planned: `health-review`, `money-advice`, `tune-tone`.)
