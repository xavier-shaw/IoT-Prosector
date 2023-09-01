import React from "react";
import { useState } from "react";
import { Button, Paper, TextField, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from "@mui/material";
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import AddCircleIcon from '@mui/icons-material/AddCircle';
import { memo } from "react";

function InstructionTable(props) {
    const { instructions, setInstructions } = props;
    const [newRow, setNewRow] = useState({
        function: '',
        interaction: '',
        image: ''
    });
    const [imagePreview, setImagePreview] = useState(null);
    const [editingCell, setEditingCell] = useState(null); // { rowIndex: null, columnName: null }

    const handleCellClick = (rowIndex, columnName) => {
        setEditingCell({ index: rowIndex, columnName: columnName });
    }

    const handleCellBlur = () => {
        setEditingCell(null);
    }

    const handleInputChange = (e, index) => {
        const { name, value } = e.target;
        if (index !== undefined) {
            const newInstructions = [...instructions];
            newInstructions[index][name] = value;
            setInstructions(newInstructions);
        } else {
            setNewRow(prevState => ({
                ...prevState,
                [name]: value
            }));
        }
    }

    const handleImageChange = (e, index) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                if (index !== undefined) {
                    const newInstructions = [...instructions];
                    newInstructions[index].image = reader.result;
                    setInstructions(newInstructions);
                } else {
                    setImagePreview(reader.result);
                    setNewRow(prevState => ({
                        ...prevState,
                        image: reader.result
                    }));
                }
            }
            reader.readAsDataURL(file);
        }
    }

    const handleAddRow = () => {
        if (!newRow.function.trim() || !newRow.interaction.trim()) {
            alert('Function and Interaction cannot be empty!');
            return;
        }

        setInstructions([...instructions, newRow]);
        setNewRow({
            function: '',
            interaction: '',
            image: ''
        });
        setImagePreview(null);
    }

    const handleDeleteRow = (index) => {
        const updatedInstructions = [...instructions];
        updatedInstructions.splice(index, 1);
        setInstructions(updatedInstructions);
    }

    return (
        <TableContainer component={Paper} sx={{ maxWidth: "100%" }}>
            <Table stickyHeader>
                <TableHead>
                    <TableRow>
                        <TableCell align="center" sx={{ fontWeight: "bold" }}>Function</TableCell>
                        <TableCell align="center" sx={{ fontWeight: "bold" }}>Interaction</TableCell>
                        <TableCell align="center" sx={{ fontWeight: "bold" }}>Image</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {instructions.map((instruction, index) => (
                        <TableRow key={index} >
                            <TableCell onClick={() => handleCellClick(index, 'function')}>
                                {editingCell?.index === index && editingCell?.columnName === 'function' ?
                                    <TextField
                                        name="function"
                                        size="small"
                                        value={instruction.function}
                                        onChange={(e) => handleInputChange(e, index)}
                                        onBlur={handleCellBlur}
                                        autoFocus
                                    /> :
                                    instruction.function
                                }
                            </TableCell>
                            <TableCell onClick={() => handleCellClick(index, 'interaction')}>
                                {editingCell?.index === index && editingCell?.columnName === 'interaction' ?
                                    <TextField
                                        name="interaction"
                                        size="small"
                                        value={instruction.interaction}
                                        onChange={(e) => handleInputChange(e, index)}
                                        onBlur={handleCellBlur}
                                        autoFocus
                                    /> :
                                    instruction.interaction
                                }
                            </TableCell>
                            <TableCell>
                                <input
                                    style={{ display: 'none' }} // This hides the default file input
                                    id={`file-upload-${index}`} // unique id for each input
                                    type="file"
                                    accept="image/*"
                                    onChange={e => handleImageChange(e, index)}
                                />
                                {instruction.image ? (
                                    <label htmlFor={`file-upload-${index}`}>
                                        {/* Clicking on this image will now trigger the file input */}
                                        <img src={instruction.image} alt="Uploaded" width="100" height="100" style={{ cursor: 'pointer' }} />
                                    </label>
                                ) : (
                                    <label htmlFor={`file-upload-${index}`}>
                                        <Button component="span" color="primary" startIcon={<CloudUploadIcon />}/>
                                    </label>
                                )}
                            </TableCell>
                            <TableCell>
                                <Button color="error" onClick={() => handleDeleteRow(index)} startIcon={<DeleteIcon />} />
                            </TableCell>
                        </TableRow>
                    ))}

                    <TableRow>
                        <TableCell>
                            <TextField
                                name="function"
                                size="small"
                                value={newRow.function}
                                onChange={handleInputChange}
                                placeholder="Function"
                            />
                        </TableCell>
                        <TableCell>
                            <TextField
                                name="interaction"
                                size="small"
                                value={newRow.interaction}
                                onChange={handleInputChange}
                                placeholder="Interaction"
                            />
                        </TableCell>
                        <TableCell>
                            <input
                                style={{ display: 'none' }}
                                id="new-file-upload"
                                type="file"
                                accept="image/*"
                                onChange={handleImageChange}
                            />
                            {imagePreview ? (
                                <label htmlFor="new-file-upload">
                                    <img src={imagePreview} alt="Preview" width="100" height="100" style={{ cursor: 'pointer' }} />
                                </label>
                            ) : (
                                <label htmlFor="new-file-upload">
                                    <Button component="span"  color="primary" startIcon={<CloudUploadIcon />} />
                                </label>
                            )}
                        </TableCell>
                        <TableCell>
                            <Button onClick={handleAddRow} startIcon={<AddCircleIcon />} />
                        </TableCell>
                    </TableRow>
                </TableBody>
            </Table>
        </TableContainer>
    )
}

export default memo(InstructionTable);