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
    let { title, onSave, onTitleChange, isSensing, handleClickExplore, updateAnnotation } = props;
    
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
                            onClick={onSave}
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
                                color={isSensing === -1? "primary" : "secondary"}
                                endIcon={<PlayArrowIcon />}
                                onClick={handleClickExplore}
                            >
                                {isSensing === -1? "Explore": "End Explore"}
                            </Button>
                            <Button
                                variant="contained"
                                color="primary"
                                endIcon={<PlayArrowIcon />}
                                onClick={updateAnnotation}
                            >
                                Annotate
                            </Button>
                            {/* <Button
                                variant="contained"
                                color="primary"
                                endIcon={<PlayArrowIcon />}
                            >
                                Verify
                            </Button> */}
                        </Stack>
                    </div>
                </Toolbar>
            </AppBar>
        </div>
    );
}
