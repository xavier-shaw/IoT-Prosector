import { useCallback, useState } from 'react';
import { Handle, Position } from 'reactflow';

export default function ExploreNode(props) {
  let { data } = props;

  return (
    <>
      <Handle type="target" position={Position.Left} />
      <h5 style={{ fontWeight: 'bold' }}>{data.label}</h5>
      <Handle type="source" position={Position.Right} />
    </>
  );
}