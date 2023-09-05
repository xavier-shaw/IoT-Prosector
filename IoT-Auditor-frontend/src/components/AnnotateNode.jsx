import { useState } from 'react';
import { Handle, Position } from 'reactflow';
import { TextField } from '@mui/material';

export default function AnnotateNode(props) {
  let { id, data } = props;
  const l = data.label;
  const [stateName, setStateName] = useState(l);
  const [editable, setEditable] = useState(false);

  const onChange = (event) => {
    data.label = event.target.value;
    setStateName(event.target.value);
  }

  return (
    <div className='nodrag'>
      <Handle type="target" position={Position.Left} />
      {editable ?
        <TextField size='small' value={stateName} autoFocus onChange={onChange} onBlur={() => { setEditable(false) }} />
        :
        <h5 style={{ fontWeight: 'bold' }} onClick={() => { setEditable(true) }}>{stateName}</h5>
      }
      <Handle type="source" position={Position.Right} />
    </div>
  );
}