# Inspiration notes — 3D reference sites

Studied 2026-06-02 for the set-visualizer. Two Three.js showcases:
Coastal World (merci-michel) and Choo-Choo World. Goal: borrow the
*feel* and proven interaction patterns, not the asset budget.

## What to adapt (ranked by value for our MVP)

### 1. Run-the-path Play button  [Choo-Choo]
A big, tactile, single-accent circular Play button floating at the screen
edge runs the train along the built track. This is exactly our "play the
set": the ball flies along the arc. Make it the most prominent control,
glowing accent. Pause = same button toggles.

### 2. URL-encoded shareable state  [Choo-Choo]
Choo-Choo encodes the whole layout in the URL (`?DATA=...`). Zero backend,
instantly shareable. ADOPT THIS: encode a set (setter pos, ball dest, peak,
speed) into the URL. Two wins at once:
- "Share it so any setter can use it" with no server.
- Dawson can text a hitter a link to their *exact* set.
Pairs with localStorage collections (your private library) vs URL (share one).

### 3. Orbit camera + snap views  [Choo-Choo]
Left-drag rotate, wheel zoom, and a compass button that snaps the camera by
45°. Validates our plan: free OrbitControls PLUS explicit Front / Side snap
buttons. When switching views, TWEEN the camera (smooth move), don't cut —
the animated camera is what reads premium.

### 4. Bottom toolbar of tactile pills  [Choo-Choo]
Bottom-center row of rounded icon buttons for pieces/tools (straight, curve,
ramp, color, undo, delete). ADOPT for our 9-man set presets: a pill row for
4/go, 31, t-ball, fast/hook, 5/red that loads a known arc shape. Makes the
demo pop and is authentic. Promote presets from phase 2 to phase 1.5.

### 5. Skippable step-by-step coach  [Choo-Choo]
First-run modal teaches one control per step (Move, Rotate, Zoom, Place,
Play) with pagination dots + Skip. Our drag interactions aren't
discoverable, so include a lightweight, skippable coach: "drag the setter,
drag the peak, hit play, switch views."

### 6. Edit/Play mode toggle  [Choo-Choo]
Bottom-right toggle flips build vs ride. Maps to our Edit vs Play (or a
clean Front/Side switch). Keep it one obvious switch.

### 7. Life and personality  [Coastal World]
The avatar has expressive state animations (idle sway, "thinking" pose,
emotions) and the world breathes (water shimmer, drifting clouds/boats).
For us, cheap touches that sell it:
- ball gets a motion trail in flight + a small squash/"pop" on landing
- a soft pulsing target ring at the ball destination
- subtle idle on the setter capsule; net tape catches a faint highlight
Whimsy without clutter (aligns with house style).

### 8. Rendering feel  [both]
Low-poly, soft/flat shading, soft contact shadows, gradient backdrop with
atmospheric haze at distance, one bold accent for the primary action,
big friendly wordmark on entry. Translate into the SetOptics dark cobalt
palette: dark studio bg (#0a0a0f), subtle gradient + vignette, glowing
cobalt/accent (#0761b2 / #4f9cf7) for the arc and Play button, soft ground
shadow, minimal floating glass chrome (top-right: title, Share, settings).

## What NOT to copy
- The long narrative onboarding funnel (Coastal) — we want straight-to-tool.
- Heavy bespoke 3D assets — keep geometry minimal (court plane, net, capsule
  setter, sphere ball, bezier arc line). The polish is in shading, camera
  tweens, the play animation, and the share, not asset count.

## Net effect on the PRD
- Promote: 9-man set presets (toolbar) and URL-share into phase 1/1.5.
- Add: animated camera tweens between Front/Side, skippable coach overlay,
  ball trail + landing pop + target ring.
- Keep: localStorage collections, two-view editing, SetOptics dark brand.
