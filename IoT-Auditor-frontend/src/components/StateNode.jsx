import { useCallback, useState } from 'react';
import { Handle, Position } from 'reactflow';
import "./StateNode.css";
import { TextField } from '@mui/material';

export default function StateNode(props) {
  let { data } = props;
  const l = data.label;
  let [stateName, setStateName] = useState(l);

  const onChange = (event) => {
    setStateName(event.target.value);
  }

  return (
    <div className="state-node" id={"state-" + data.id}>
      <Handle type="target" position={Position.Top} />
      <div>
        <TextField label="State" value={stateName} onChange={onChange} />
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}