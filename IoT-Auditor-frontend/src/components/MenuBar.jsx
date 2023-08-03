import * as React from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Button from '@mui/material/Button';
import MenuItem from '@mui/material/MenuItem';
import Menu from '@mui/material/Menu';
import ReactLogo from '../privacy_logo.svg';
import EditableLabel from 'react-inline-editing';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { Stack } from '@mui/material';

export default function MenuBar(props) {
    let { title, onMenuClick, onTitleChange, runProgram } = props;
    const [anchorEl, setAnchorEl] = React.useState(null);
    const open = Boolean(anchorEl);

    const handleClick = (event) => {
        setAnchorEl(event.currentTarget);
    };

    const handleClose = (event) => {
        setAnchorEl(null);

        switch (event) {
            case "save":
                onMenuClick("save");
                break;
            case "duplicate":
                onMenuClick("duplicate");
            default:
                break;
        }
    };

    return (
        <div>
            <AppBar sx={{ height: "60px", display: "flex" }} color="transparent" position="static">
                <Toolbar>
                    <div className="boardlogo">
                        <a href="/"><img src={ReactLogo} alt="React Logo" /> IoT Auditor</a>
                    </div>
                    <div className='ms-4'>
                        <Button
                            variant="contained"
                            color="primary"
                            endIcon={<PlayArrowIcon />}
                            onClick={runProgram}
                        >
                            Save
                        </Button>
                    </div>
                    <div className="ms-auto">
                        {title ? (
                            <EditableLabel
                                text={title}
                                labelPlaceHolder="untitled"
                                labelClassName='titleLable'
                                inputClassName='titleInput'
                                inputWidth='300px'
                                inputHeight='25px'
                                inputMaxLength={50}
                                labelFontWeight='normal'
                                inputFontWeight='normal'
                                onFocusOut={onTitleChange}
                            />
                        ) : (
                            <div></div>
                        )}
                    </div>
                    <div className='ms-auto'>
                        <Stack spacing={2} direction={"row"}>
                            <Button
                                variant="contained"
                                color="primary"
                                endIcon={<PlayArrowIcon />}
                                onClick={runProgram}
                            >
                                Explore
                            </Button>
                            <Button
                                variant="contained"
                                color="primary"
                                endIcon={<PlayArrowIcon />}
                                onClick={runProgram}
                            >
                                Annotate
                            </Button>
                            <Button
                                variant="contained"
                                color="primary"
                                endIcon={<PlayArrowIcon />}
                                onClick={runProgram}
                            >
                                Verify
                            </Button>
                        </Stack>
                    </div>
                </Toolbar>
            </AppBar>
        </div>
    );
}
