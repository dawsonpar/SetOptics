export function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

export function smoothStep(t: number) {
  const clamped = clamp01(t);
  return clamped * clamped * (3 - 2 * clamped);
}
