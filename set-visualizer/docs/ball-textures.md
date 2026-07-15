# Ball textures

The volleyball is a plain `sphereGeometry` with authored equirectangular
maps; there is no 3D model. Two balls ship, selected by format in
`src/lib/balls.ts` (`FORMAT_BALL`):

| Ball | Format | Look |
|------|--------|------|
| Molten V5M5000 (Flistatec) | 9-man | white base, paired red/blue swooshes, hex emboss |
| Mikasa V200W | indoor 6s | yellow base, blue swirl bands, dimple emboss |

Both are procedural recreations referenced from photos of the physical
balls. Logos are omitted by construction (the generator only draws panel
geometry, never marks).

## Files

- `public/textures/balls/<ball>_albedo.webp` — 2048x1024 color map (sRGB)
- `public/textures/balls/<ball>_normal.webp` — 512x256 normal map
  (seam grooves + surface emboss; the ball renders small, so low res is fine)

## Regenerating

```
cd tools/ball-textures
python3 -m venv venv && ./venv/bin/pip install numpy pillow
./venv/bin/python gen_ball_textures.py out ball_params.json
```

Then convert to webp (albedo q82 full size, normal resized to 512x256 q80)
and copy into `public/textures/balls/`. Pattern parameters (band counts,
swirl, widths, colors, emboss strength) all live in `ball_params.json`;
the swirl model is documented at the top of `gen_ball_textures.py`.

Rendering notes: `Ball.tsx` sets `albedo.colorSpace = SRGBColorSpace`,
uses the albedo as a low-intensity emissive map so the ball stays readable
against the dark stage, and spins the inner mesh about the axis
perpendicular to travel while playing (squash stays on the outer group so
it remains world-vertical).
