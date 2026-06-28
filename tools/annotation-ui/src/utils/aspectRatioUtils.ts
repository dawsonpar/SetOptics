export type AspectRatio = '16:9' | '9:16' | '1:1' | '4:3' | '4:5';

export function getAspectRatioValue(aspectRatio: AspectRatio): number {
  switch (aspectRatio) {
    case '16:9': return 16 / 9;
    case '9:16': return 9 / 16;
    case '1:1':  return 1;
    case '4:3':  return 4 / 3;
    case '4:5':  return 4 / 5;
  }
}

export function getAspectRatioDimensions(
  aspectRatio: AspectRatio,
  baseWidth: number
): { width: number; height: number } {
  const ratio = getAspectRatioValue(aspectRatio);
  return {
    width: baseWidth,
    height: baseWidth / ratio,
  };
}

export function getAspectRatioLabel(aspectRatio: AspectRatio): string {
  return aspectRatio;
}


export function formatAspectRatioForCSS(aspectRatio: AspectRatio): string {
  return aspectRatio.replace(':', '/');
}