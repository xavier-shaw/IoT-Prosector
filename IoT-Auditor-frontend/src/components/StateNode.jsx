import { useCallback, useState } from 'react';
import { Handle, Position } from 'reactflow';
import "./StateNode.css";
import { TextField } from '@mui/material';

const handleStyle = { left: 10 };

export default function StateNode(props) {
  let { data } = props;
  let [stateName, setStateName] = useState(data.label);

  const onChange = (event) => {
    setStateName(event.target.value);
    data.label = event.target.value;
  }

  return (
    <div className="state-node">
      <Handle type="target" position={Position.Top} />
      <div>
        <TextField label="State" value={stateName} onChange={onChange} />
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        id="a"
        style={handleStyle}
      />
      <Handle type="source" position={Position.Bottom} id="b" />
    </div>
  );
}