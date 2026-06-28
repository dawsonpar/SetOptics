import type { ZoomRegion } from "../types";
import { smoothStep } from "./mathUtils";
import { TRANSITION_WINDOW_MS } from "./constants";

export function computeRegionStrength(region: ZoomRegion, timeMs: number) {
  const leadInStart = region.startMs - TRANSITION_WINDOW_MS;
  const leadOutEnd = region.endMs + TRANSITION_WINDOW_MS;

  if (timeMs < leadInStart || timeMs > leadOutEnd) {
    return 0;
  }

  const fadeIn = smoothStep((timeMs - leadInStart) / TRANSITION_WINDOW_MS);
  const fadeOut = smoothStep((leadOutEnd - timeMs) / TRANSITION_WINDOW_MS);
  return Math.min(fadeIn, fadeOut);
}

export function findDominantRegion(regions: ZoomRegion[], timeMs: number) {
  let bestRegion: ZoomRegion | null = null;
  let bestStrength = 0;

  for (const region of regions) {
    const strength = computeRegionStrength(region, timeMs);
    if (strength > bestStrength) {
      bestStrength = strength;
      bestRegion = region;
    }
  }

  return { region: bestRegion, strength: bestStrength };
}
