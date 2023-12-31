import { useCallback, useState } from 'react';
import { Handle, NodeResizeControl, Position } from 'reactflow';
import { TextField } from '@mui/material';
import { groupZIndex, childNodeMarginY, childNodeoffsetY } from '../shared/chartStyle';

export default function SemanticNode(props) {
    let { id, data } = props;
    const l = data.label;
    const [editable, setEditable] = useState(false);
    const [nodeName, setNodeName] = useState(l);

    const onChange = (event) => {
        data.label = event.target.value;
        setNodeName(event.target.value);
    };

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
        <div style={{ zIndex: groupZIndex }}>
            <NodeResizeControl controlStyle={controlStyle}>
                <ResizeIcon />
            </NodeResizeControl>
            {data.children.map((child, idx) => { 
                return (
                <Handle key={idx} type="target" id={"target-" + idx} position={Position.Left}
                    style={{ top: childNodeMarginY + idx * childNodeoffsetY + 37 }} />
            )})}
            {/* {editable ?
                <TextField value={nodeName} autoFocus onChange={onChange} onBlur={() => { setEditable(false) }} className='nodrag'/>
                :
                <h4 style={{fontWeight: 'bold'}} onClick={() => { setEditable(true) }}>{nodeName}</h4>
            } */}
            {data.children.map((child, idx) => (
                    <Handle key={idx} type="source" id={"source-" + idx} position={Position.Right} 
                    style={{ top: childNodeMarginY + idx * childNodeoffsetY + 37 }} />
                )
            )}
        </div>
    );
}