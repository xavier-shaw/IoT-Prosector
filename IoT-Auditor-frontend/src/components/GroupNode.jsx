import { useState } from 'react';
import { Handle, Position } from 'reactflow';
import { Divider, TextField } from '@mui/material';

export default function GroupNode(props) {
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
                <TextField size='small' label="Group" value={stateName} autoFocus onChange={onChange} onBlur={() => { setEditable(false) }} className='nodrag' />
                :
                <h4 onClick={() => { setEditable(true) }}>{stateName}</h4>
            }
            {(() => (data.subNodes.map((subNode) => (
                    <h5 key={subNode.id}>{subNode.data.label}</h5>
                )))
                )()}
            <Handle type="source" position={Position.Bottom} />
        </>
    );
}