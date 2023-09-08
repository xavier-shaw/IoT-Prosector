import { useCallback, useState } from 'react';
import { Handle, NodeResizeControl, Position } from 'reactflow';
import { TextField } from '@mui/material';
import { groupZIndex, childNodeMarginY, childNodeoffsetY, displayHandleMargin, displayHandleOffset } from '../shared/chartStyle';

export default function DisplayNode(props) {
    let { id, data } = props;
    const l = data.representative;
    const [editable, setEditable] = useState(false);
    const [nodeName, setNodeName] = useState(l);

    const onChange = (event) => {
        data.representative = event.target.value;
        setNodeName(event.target.value);
    };

    return (
        <div style={{ zIndex: groupZIndex }}>
            {data.children && data.children?.map((child, idx) => (
                <Handle key={idx} type="target" id={"target-" + child} position={Position.Left}
                    style={{ top: displayHandleMargin + idx * displayHandleOffset }} />
            ))}
            {!data.children && <Handle type="target" position={Position.Left} />}
            {editable ?
                <TextField className="m-auto nodrag" value={nodeName} autoFocus onChange={onChange} onBlur={() => { setEditable(false) }} />
                :
                <h5 className='m-auto' style={{ fontWeight: 'bold' }} onClick={() => { setEditable(true) }}>{nodeName? nodeName: data.representative}</h5>
            }
            {data.children && data.children?.map((child, idx) => (
                    <Handle key={idx} type="source" id={"source-" + child} position={Position.Right} 
                    style={{ top: displayHandleMargin + idx * displayHandleOffset }} />
                )
            )}
            {!data.children && <Handle type="source" position={Position.Right} />}
        </div>
    );
}