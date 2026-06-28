import { ZOOM_DEPTH_SCALES, clampFocusToDepth, type ZoomFocus, type ZoomDepth } from "../types";

interface StageSize {
  width: number;
  height: number;
}

export function clampFocusToStage(
  focus: ZoomFocus,
  depth: ZoomDepth,
  stageSize: StageSize
): ZoomFocus {
  if (!stageSize.width || !stageSize.height) {
    return clampFocusToDepth(focus, depth);
  }

  const zoomScale = ZOOM_DEPTH_SCALES[depth];
  
  const windowWidth = stageSize.width / zoomScale;
  const windowHeight = stageSize.height / zoomScale;
  
  const marginX = windowWidth / (2 * stageSize.width);
  const marginY = windowHeight / (2 * stageSize.height);

  const baseFocus = clampFocusToDepth(focus, depth);

  return {
    cx: Math.max(marginX, Math.min(1 - marginX, baseFocus.cx)),
    cy: Math.max(marginY, Math.min(1 - marginY, baseFocus.cy)),
  };
}

export function stageFocusToVideoSpace(
  focus: ZoomFocus,
  stageSize: StageSize,
  videoSize: { width: number; height: number },
  baseScale: number,
  baseOffset: { x: number; y: number }
): ZoomFocus {
  if (!stageSize.width || !stageSize.height || !videoSize.width || !videoSize.height || baseScale <= 0) {
    return focus;
  }

  const stageX = focus.cx * stageSize.width;
  const stageY = focus.cy * stageSize.height;

  const videoNormX = (stageX - baseOffset.x) / (videoSize.width * baseScale);
  const videoNormY = (stageY - baseOffset.y) / (videoSize.height * baseScale);

  return {
    cx: videoNormX,
    cy: videoNormY,
  };
}
