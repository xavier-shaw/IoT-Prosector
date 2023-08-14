import { useCallback, useState } from 'react';
import { Handle, Position } from 'reactflow';

export default function ExploreNode(props) {
  let { data } = props;

  return (
    <>
      <Handle type="target" position={Position.Top} />
      <h5 className='nodrag'>{data.label}</h5>
      <Handle type="source" position={Position.Bottom} />
    </>
  );
}