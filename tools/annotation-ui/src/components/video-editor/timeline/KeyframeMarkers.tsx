import React from "react";
import { useTimelineContext } from "dnd-timeline";

interface Keyframe {
  id: string;
  time: number;
}

interface KeyframeMarkersProps {
  keyframes: Keyframe[];
  selectedKeyframeId: string | null;
  setSelectedKeyframeId: (id: string | null) => void;
}

const KeyframeMarkers: React.FC<KeyframeMarkersProps> = ({ keyframes, selectedKeyframeId, setSelectedKeyframeId }) => {
  const { sidebarWidth, range, valueToPixels } = useTimelineContext();
  return (
    <>
      {keyframes.map(kf => {
        const offset = valueToPixels(kf.time - range.start);
        const isSelected = kf.id === selectedKeyframeId;
        return (
          <div
            key={kf.id}
            className={`absolute top-8 cursor-pointer ${isSelected ? 'ring-2 ring-[#34B27B]' : ''}`}
            style={{ left: `${sidebarWidth + offset - 8}px`, zIndex: 40 }}
            onClick={e => {
              e.stopPropagation();
              setSelectedKeyframeId(kf.id);
            }}
            title={`Keyframe @ ${kf.time}ms`}
          >
            <div style={{
              width: '10px',
              height: '10px',
              background: '#ffe100ff',
              transform: 'rotate(45deg)',
              border: 'none',
             
              opacity: isSelected ? 1 : 0.6,
              transition: 'opacity 0.15s',
            }} />
          </div>
        );
      })}
    </>
  );
};

export default KeyframeMarkers;
