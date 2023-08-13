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
    <>
      <Handle type="target" position={Position.Top} />
      {editable ?
        <TextField size='small' label="State" value={stateName} autoFocus onChange={onChange} onBlur={() => { setEditable(false) }} className='nodrag' />
        :
        <h5 onClick={() => { setEditable(true) }}>{stateName}</h5>
      }
      <Handle type="source" position={Position.Bottom} />
    </>
  );
}