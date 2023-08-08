import { useCallback, useState } from 'react';
import { Handle, Position } from 'reactflow';
import "./StateNode.css";
import { TextField } from '@mui/material';

export default function ExploreNode(props) {
  let { data } = props;

  return (
    <div className="state-node" id={"state-" + data.id}>
      <Handle type="target" position={Position.Top} />
      <div>
        <TextField label="State" disabled value={data.label}/>
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}