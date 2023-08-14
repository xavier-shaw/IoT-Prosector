import { useCallback, useState } from 'react';
import { Handle, Position } from 'reactflow';
import { TextField } from '@mui/material';

export default function ExploreNode(props) {
  let { data } = props;

  return (
    <>
      <Handle type="target" position={Position.Left} />
      <h5 className='nodrag'>{data.label}</h5>
      <Handle type="source" position={Position.Right} />
    </>
  );
}