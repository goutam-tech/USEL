---
name: Precision Technical Lab
colors:
  surface: '#fcf8f8'
  surface-dim: '#dcd9d9'
  surface-bright: '#fcf8f8'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f6f3f2'
  surface-container: '#f0edec'
  surface-container-high: '#ebe7e7'
  surface-container-highest: '#e5e2e1'
  on-surface: '#1c1b1b'
  on-surface-variant: '#3f493f'
  inverse-surface: '#313030'
  inverse-on-surface: '#f3f0ef'
  outline: '#6f7a6e'
  outline-variant: '#becabc'
  surface-tint: '#006d30'
  primary: '#00652c'
  on-primary: '#ffffff'
  primary-container: '#15803d'
  on-primary-container: '#d3ffd5'
  inverse-primary: '#79db8d'
  secondary: '#5f5e61'
  on-secondary: '#ffffff'
  secondary-container: '#e4e1e6'
  on-secondary-container: '#656467'
  tertiary: '#00652b'
  on-tertiary: '#ffffff'
  tertiary-container: '#008138'
  on-tertiary-container: '#d5ffd5'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#95f8a7'
  primary-fixed-dim: '#79db8d'
  on-primary-fixed: '#00210a'
  on-primary-fixed-variant: '#005323'
  secondary-fixed: '#e4e1e6'
  secondary-fixed-dim: '#c8c5ca'
  on-secondary-fixed: '#1b1b1e'
  on-secondary-fixed-variant: '#47464a'
  tertiary-fixed: '#6bff8f'
  tertiary-fixed-dim: '#4ae176'
  on-tertiary-fixed: '#002109'
  on-tertiary-fixed-variant: '#005321'
  background: '#fcf8f8'
  on-background: '#1c1b1b'
  surface-variant: '#e5e2e1'
typography:
  display-hero:
    fontFamily: Geist
    fontSize: 48px
    fontWeight: '600'
    lineHeight: 52px
    letterSpacing: -0.04em
  display-hero-mobile:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 38px
    letterSpacing: -0.03em
  headline-lg:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 38px
    letterSpacing: -0.025em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 30px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '500'
    lineHeight: 30px
    letterSpacing: -0.02em
  headline-sm:
    fontFamily: Geist
    fontSize: 18px
    fontWeight: '500'
    lineHeight: 24px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: -0.011em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: -0.006em
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0em
  mono-code:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: -0.01em
  mono-inline:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: -0.01em
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.06em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  space-2xs: 0.125rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-base: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
  space-2xl: 3rem
  gutter-mobile: 1rem
  gutter-desktop: 1.5rem
  max-width: 80rem
---

## Brand & Style

This design system delivers a clinical, hyper-precise aesthetic built for engineers, researchers, and technical practitioners who demand absolute signal with zero decorative noise. Drawing from modern technical minimalism and Swiss typographic order, the visual tone balances raw computational efficiency with uncompromising craftsmanship. 

The aesthetic is characterized by stark architectural whites, ink-dense black typography, 1px structural hairline divisions, and deliberate punctures of a razor-sharp electric kinetic green. It evokes the feeling of high-end developer instrumentation, quantum computing dashboards, and specialized physics lab hardware: rigorous, predictable, and exceptionally responsive.

## Colors

The palette is engineered around high contrast, functional clarity, and deliberate focal guidance.

- **Canvas & Surfaces:** Primary workspace is pure clinical white (`#FFFFFF`). Secondary card layers and code panel backgrounds use `#F8FAFC`, with sunken wells, inactive table headers, and track backgrounds set to `#F1F5F9`.
- **Text & Structure:** Core text uses deep obsidian `#0A0A0A` for primary reading hierarchy, stepping down to `#18181B` for secondary content, and `#64748B` for tertiary metadata. Structural dividers and grid strokes are confined to ultra-fine 1px `#E2E8F0`.
- **Accent & State:** The accent system employs a dual-tone green strategy: deep scientific forest `#15803D` provides WCAG AAA compliant text, active indicators, and solid button surfaces against pure white, while electric kinetic lime `#22C55E` serves as an emissive accent for micro-badges, status pings, cursor lines, and dark-surface code highlights.

## Typography

The typographic system relies on three complementary typefaces executing specific functional roles:

- **Geist** delivers crisp geometric structure for titles, headers, and dashboard metrics, utilizing negative tracking to maintain tight optical density.
- **Inter** handles narrative readability, forms, and dense data layouts with neutral clarity.
- **JetBrains Mono** governs terminal output, data payloads, numerical values, telemetry readouts, status badges, and table headers.

Keep tabular numbers (`font-variant-numeric: tabular-nums`) enabled across all numeric data displays and monospace metrics.

## Layout & Spacing

The layout is governed by a strict 4px/8px modular baseline grid. Visual hierarchy is established via proportional spatial density rather than heavy decorative padding.

- **Desktop (1024px+):** 12-column fluid grid, 24px gutters, max-width bounded at 1280px. High-density panels utilize compact 12px interior padding.
- **Tablet (768px - 1023px):** 8-column layout with 16px gutters and persistent modular toolbars.
- **Mobile (<768px):** 4-column layout with 16px screen-edge margins. Multi-column data collapses into vertical stacked cards or horizontally scrollable data lanes.

## Elevation & Depth

This design system avoids blurred, heavy drop shadows in favor of **Tonal Layering** and **Fine Hairline Outlines**.

1. **Layer 0 (Base Canvas):** `#FFFFFF` canvas for top-level pages and documentation.
2. **Layer 1 (Card & Module Surfaces):** `#FFFFFF` or `#F8FAFC` defined by a solid `1px solid #E2E8F0` border.
3. **Layer 2 (Floating Overlays & Menus):** Pure `#FFFFFF` surface framed with `1px solid #CBD5E1` combined with a technical micro-shadow: `0 1px 2px 0 rgba(0, 0, 0, 0.05), 0 4px 12px 0 rgba(0, 0, 0, 0.03)`.
4. **Interactive Focus:** Keyboard and click focus states use a sharp, zero-blur outline: `0 0 0 1px #FFFFFF, 0 0 0 3px #15803D`.

## Shapes

The shape system is intentionally subtle and disciplined (`roundedness: 1`). Radii are kept small to preserve an architectural, modular feel.

- Micro-elements (badges, inline code tags, checkboxes): `2px` to `4px`.
- Standard interactive elements (buttons, inputs, select fields): `4px` (`0.25rem`).
- Container panels, cards, and modal dialogs: `6px` to `8px` (`0.375rem` to `0.5rem`).
- Strict rule: Fully circular pill shapes are disallowed, with the sole exception of pulsing 6px/8px live status dots.

## Components

### Buttons
- **Primary:** Solid `#0A0A0A` background with `#FFFFFF` text. On hover, shifts to `#18181B`. When executing live actions, shifts to `#15803D`.
- **Accent Variant:** `#15803D` background with `#FFFFFF` text. Hover transitions to `#166534`.
- **Secondary / Outline:** `#FFFFFF` background, `1px solid #E2E8F0`, `#0A0A0A` text. Hover introduces `#F8FAFC` background and `#CBD5E1` border.
- **Ghost:** Transparent background with `#0A0A0A` text. Hover yields `#F1F5F9`.

### Input Fields & Selects
- Constructed on a `#FFFFFF` base, framed by a `1px solid #E2E8F0` border, `4px` corner radius, with 8px horizontal padding.
- Text rendered in `Inter` 14px (`#0A0A0A`), placeholder text in `#94A3B8`.
- Focus state activates an instantaneous `#15803D` border with a 1px matching ring.

### Checkboxes & Radio Buttons
- Precision square (`4px` radius) or circle, `16px x 16px`, bordered with `1px solid #CBD5E1`.
- Checked state fills with `#15803D`, displaying an unaliased crisp white check or center pip.

### Chips & Badges
- Heights restricted to 20px–24px. Built with `JetBrains Mono` 11px uppercase.
- Neutral: `#F1F5F9` background, `#475569` text, `1px solid #E2E8F0`.
- Active / Success: `#DCFCE7` background, `#15803D` text, `1px solid #BBF7D0`. Accompanied by a 6px circular `#22C55E` status pip.

### Cards & Data Panels
- Background `#FFFFFF` with `1px solid #E2E8F0` framing.
- Header bars utilize `#F8FAFC` with a bottom `1px solid #E2E8F0` border and `JetBrains Mono` label metadata.

### Code Blocks & Telemetry Terminals
- Background `#0A0A0A` or `#F8FAFC` (for light syntax).
- Light code panels use `1px solid #E2E8F0`, syntax highlighted with `#15803D` keywords, `#0A0A0A` variables, and `#64748B` line numbers.