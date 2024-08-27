import { useCallback, useState } from 'react';
import { Handle, Position } from 'reactflow';
import { childNodeMarginY, childNodeoffsetY, stateZIndex } from '../shared/chartStyle';

export default function ExploreNode(props) {
  let { data } = props;

  return (
    <div style={{
      display: "flex",
      justifyContent: "center",
      alignItems: "center", zIndex: stateZIndex
    }}>
      <Handle type="target" position={Position.Left} />
      <p className="m-auto" style={{ fontFamily: "Times New Roman", fontSize: 30, fontWeight: 'bold' }}>{data.label}</p>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}