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
    let { title, onSave, saved, onTitleChange, step, handleClickNext, handleClickBack, annotated} = props;

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
                            disabled={step == 2 || annotated === 1}
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
                                labelFontWeight='bold'
                                inputFontWeight='normal'
                                onFocusOut={onTitleChange}
                            />
                        ) : (
                            <div></div>
                        )}
                    </div>
                    <div className='ms-auto'>
                        <Stack spacing={2} direction={"row"}>
                            {/* <Button
                                variant="outlined"
                                color="error"
                                endIcon={<PlayArrowIcon />}
                                disabled={step === 0}
                                onClick={handleClickBack}
                            >
                                Back
                            </Button> */}
                            <Button
                                variant="contained"
                                color="primary"
                                endIcon={<PlayArrowIcon />}
                                onClick={handleClickNext}
                                disabled={!saved || (step === 1 && annotated !== 2)}
                            >
                                {step !== 2 && "Next"}
                                {step === 2 && "Finish"}
                            </Button>
                        </Stack>
                    </div>
                </Toolbar>
            </AppBar>
        </div>
    );
}
