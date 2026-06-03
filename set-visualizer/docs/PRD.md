# Set Visualizer — PRD (Phase 1 / MVP)

Interactive 3D volleyball set visualizer. A setter places themselves and a
ball destination on the court, shapes the set's arc by dragging its peak, and
plays the set as an animation. Two snap views (front + side) make the 3D arc
fully editable. Lives in the open-source SetOptics repo, deployed to Vercel
for any setter to use.

Built as the "Build" payoff for the @dp.mp4 9-man arc (episode 2). Companion
docs: `inspiration.md` (3D reference patterns).

## Users
- Primary: Dawson, mapping 20+ hitters by saving sets into collections.
- Shared: any setter/coach opening the public link; their data is local to
  their browser; a single set is shareable by URL.

## Coordinate model (meters)
- X = lateral, along the net. Center 0, court half-width 5 (court is 10m wide).
- Y = up (height). Floor at 0.
- Z = depth from the net into the court. Net plane at Z = 0; setter's half is
  Z in [0, 10] (9-man court is 10 x 20m, one half = 10 x 10m).
- Net: plane at Z = 0 spanning X[-5,5], height [0, NET_HEIGHT].
  NET_HEIGHT default 2.43m — VERIFY against NACIVT rulebook.

## Entities (confirmed)
- Setter (origin): draggable marker on the floor (X, Z). Ball contact starts
  at setter + CONTACT_HEIGHT (~2.2m).
- Ball destination: the hitter's attack point near the net, draggable (X, Y, Z).
- Arc peak: draggable control handle. Front view drag = X + height; side view
  drag = depth (Z) + height.
- Set arc: quadratic Bezier P0=setter contact, P1=peak, P2=destination.
- Ball: sphere animating along the arc on play, with motion trail + landing pop.

## Views (camera)
- Front: camera square to the net (looks along Z). Editing plane = constant Z;
  drag moves handles in X (horizontal) and Y (vertical).
- Side: camera down the net line (looks along X). Editing plane = constant X;
  drag moves handles in Z (horizontal) and Y (vertical).
- Free orbit available (OrbitControls); Front/Side buttons snap with a tweened
  (animated) camera move, not a cut.

## Controls
- Drag setter, destination, peak (in the active view's plane).
- Play / pause (big glowing accent button) flies the ball along the arc.
- Speed slider (set tempo).
- View switch: Front / Side / Orbit.
- 9-man set presets (pill toolbar): 4/go, 31, t-ball (2), fast/hook, 5/red.
  Each loads a known destination + arc shape; user can then adjust.

## Collections (in-browser, no server)
- Collection = named folder of saved sets. Abstract: "Hitter: Marcus",
  "Sets to work on", etc.
- Save current set (setter, destination, peak, speed, optional name) into a
  collection. Create / rename / delete collections; list, load, delete sets.
- Persistence: localStorage. Each user's library is private to their browser.
- Export / import a collection as JSON (backup / move). [nice-to-have]

## Sharing (no server)
- Encode the current set (setter, destination, peak, speed) into the URL query
  (compact). Opening the link restores that exact set. Lets Dawson send a
  hitter a link to their set; lets anyone share a set.

## Aesthetics — match SetOptics
- Dark studio bg #0a0a0f with subtle gradient + vignette.
- Cobalt primary #0761b2, accent blue #4f9cf7; arc + play button glow.
- Low-poly soft court, soft contact shadow under setter/ball, faint net.
- Minimal floating glass chrome (top: wordmark, Share, settings).
- DM Sans / DM Mono for UI text + labels (house style).
- Skippable first-run coach overlay (drag setter, drag peak, play, switch views).

## Stack
- Vite + React + TypeScript + three + @react-three/fiber + @react-three/drei.
- zustand for app state. Tailwind for UI chrome (SetOptics tokens).
- localStorage persistence; URL-param sharing.
- Deploy: Vercel static SPA. Path: SetOptics/set-visualizer/.

## Scope
Phase 1 / today (demoable + filmable for the glimpse):
- Court/net, draggable setter + destination + peak, front/side/orbit views,
  play/pause, speed, SetOptics brand. PLUS presets + collections + URL-share
  (per Dawson: presets and collections are in scope today).
Phase 1.5 / before tomorrow's public share:
- Coach overlay, export/import, ball trail + landing pop polish, deploy.
Phase 2 / later (out of scope now):
- Hitting-boxes mode (adjustable boxes like the paper diagram).
- Text notes per hitter; tempo metrics; multi-set overlay compare.

## Non-goals
- No server / backend / auth. All in-browser.
- Not real aerodynamic physics; a believable Bezier arc.
- Not the paper "hitting boxes" yet (phase 2).

## Acceptance (today)
- Open app, drag setter/destination/peak in both views, press play, ball flies
  the set, adjust speed. Load a 9-man preset. Save a set into a named
  collection and reload it. Copy a share URL that restores the set. Looks
  on-brand (dark cobalt). Runs at 60fps on the laptop.

## Repo note
SetOptics is a PUBLIC repo. Per the no-public-push-during-workday rule, build
and demo locally today; push/deploy outside 9-5 ET (the public share is
tomorrow anyway).
