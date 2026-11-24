# [EPIC] Wave 2: methane inversion on TROPOMI

Draft ID: `MI-01`

## Goal

A first end-to-end methane inversion on one month of TROPOMI data.

## Theme Epics

- [ ] MI-02

## Relationships

- Blocks: none

---

# [EPIC] Observation operators

Draft ID: `MI-02`

## Theme

Everything that maps a model state to TROPOMI columns.

## Parent Wave

MI-01

## Issues

- [ ] MI-03
- [ ] MI-04

## Relationships

- Parent: MI-01

---

# feat(data): TROPOMI CH4 L2 reader

Draft ID: `MI-03`

## Problem / Request

Read TROPOMI CH4 L2 orbits into the stack's carrier.

## Relationships

- Parent: MI-02
- Blocks: MI-04

---

# feat(models): averaging-kernel observation operator

Draft ID: `MI-04`

## Problem / Request

Apply the averaging kernel and prior profile to a model column.

## Relationships

- Parent: MI-02
- Blocked by: MI-03
- Related: #188
