import { useState } from 'react';
import { Handle, NodeResizeControl, Position } from 'reactflow';
import { Chip, TextField } from '@mui/material';

export default function GroupNode(props) {
    let { id, data } = props;
    const l = data.label;
    const [stateName, setStateName] = useState(l);
    const [editable, setEditable] = useState(false);

    const onChange = (event) => {
        data.label = event.target.value;
        setStateName(event.target.value);
    }

    function ResizeIcon() {
        return (
            <svg
                xmlns="http://www.w3.org/2000/svg"
                width="20"
                height="20"
                viewBox="0 0 24 24"
                strokeWidth="2"
                stroke="#ff0071"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ position: 'absolute', right: 5, bottom: 5 }}
            >
                <path stroke="none" d="M0 0h24v24H0z" fill="none" />
                <polyline points="16 20 20 20 20 16" />
                <line x1="14" y1="14" x2="20" y2="20" />
                <polyline points="8 4 4 4 4 8" />
                <line x1="4" y1="4" x2="10" y2="10" />
            </svg>
        );
    };

    const controlStyle = {
        background: 'transparent',
        border: 'none',
    };

    return (
        <>
            <NodeResizeControl controlStyle={controlStyle}>
                <ResizeIcon />
            </NodeResizeControl>
            <Handle type="target" position={Position.Top} />
            {editable ?
                <TextField size='small' value={stateName} autoFocus onChange={onChange} onBlur={() => { setEditable(false) }} className='nodrag' />
                :
                <h4 onClick={() => { setEditable(true) }}>{stateName}</h4>
            }
            {(() => (data.subNodes.map((subNode) => (
                <Chip style={{ backgroundColor: "#788bff", fontSize: "18px", margin: "2px" }} key={subNode.id} label={subNode.data.label} />
            )))
            )()}
            <Handle type="source" position={Position.Bottom} />
        </>
    );
}