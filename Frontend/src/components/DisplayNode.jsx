import { useCallback, useState } from 'react';
import { Handle, NodeResizeControl, Position } from 'reactflow';
import { TextField } from '@mui/material';
import { groupZIndex, childNodeMarginY, childNodeoffsetY, displayHandleMargin, displayHandleOffset } from '../shared/chartStyle';

export default function DisplayNode(props) {
    let { id, data } = props;
    const l = data.label;
    const [editable, setEditable] = useState(false);
    const [nodeName, setNodeName] = useState(l);

    const onChange = (event) => {
        data.label = event.target.value;
        setNodeName(event.target.value);
    };

    let maxHandles = data.inEdgeNum > data.outEdgeNum? data.inEdgeNum: data.outEdgeNum;
    let height = displayHandleMargin + maxHandles * displayHandleOffset;

    const targetEdgeHandles = Array.from({ length: maxHandles }, (v, i) => {
        return (
            <Handle key={i} type="target" id={"in-" + i} position={Position.Left}
            style={{ top: (height / (maxHandles + 1)) * (i + 1) }} />
        )
    });

    const sourceEdgeHandles = Array.from({ length: maxHandles }, (v, i) => (
        <Handle key={i} type="source" id={"out-" + i} position={Position.Right}
            style={{ top: (height / (maxHandles + 1)) * (i + 1) }} />
    ));

    return (
        <div style={{ zIndex: groupZIndex }}>
            {targetEdgeHandles}
            {editable ?
                <TextField className="m-auto nodrag" value={nodeName} autoFocus onChange={onChange} onBlur={() => { setEditable(false) }} />
                :
                <h5 className='m-auto' style={{ fontFamily: "Times New Roman", fontSize: 30, fontWeight: 'bold' }} onClick={() => { setEditable(true) }}>{data.label}</h5>
            }
            {sourceEdgeHandles}
        </div>
    );
}