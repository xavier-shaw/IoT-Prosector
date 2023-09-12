import { useState } from 'react';
import { Handle, Position } from 'reactflow';
import { TextField } from '@mui/material';
import zIndex from '@mui/material/styles/zIndex';
import { stateZIndex } from '../shared/chartStyle';

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
    <div className='nodrag' style={{ zIndex: stateZIndex }}>
      <Handle type="target" position={Position.Left} />
      {editable ?
        <TextField className="m-auto" size='small' value={stateName} autoFocus onChange={onChange} onBlur={() => { setEditable(false) }} />
        :
        <p className="m-auto" style={{ fontFamily: "Times New Roman", fontSize: 30, fontWeight: 'bold' }} onClick={() => { setEditable(true) }}>{stateName}</p>
      }
      <Handle type="source" position={Position.Right} />
    </div>
  );
}