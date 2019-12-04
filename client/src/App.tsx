import React, { useState, useRef } from 'react';
import './App.css';
import { Button, TextField, Grid, Paper, InputLabel } from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';
import axios, { AxiosRequestConfig, AxiosResponse } from 'axios'

const useStyles = makeStyles(theme => ({
    root: {
        flexGrow: 1,
        '& > *': {
            margin: theme.spacing(1),
        },
    },
    paperHeader: {
        padding: theme.spacing(2),
        textAlign: 'center',
        color: theme.palette.text.primary,
        fontSize: 18,
        backgroundColor: '#42a1f5'
    },

    paperOut: {
        padding: theme.spacing(2),
        textAlign: 'center',
        color: theme.palette.text.primary,
        fontSize: 14,
        fontWeight: "bold"
    },
    textField: {
        marginLeft: theme.spacing(1),
        marginRight: theme.spacing(1),
        width: 600,
    },
    orange: {
        backgroundColor: '#f5aa42'
    },
    yellow: {
        backgroundColor: '#f5aa42'
    },
    blue: {
        backgroundColor: '#42cbf5'
    },
    darkBlue: {
        backgroundColor: '#42cbf5'
    },
    green: {
        backgroundColor: '#00ff00'
    },
    red: {
        backgroundColor: '#ff0000'
    }
}));

const estimateReview = async (text: string) => {

    const headers: AxiosRequestConfig['headers'] = {
        'Content-Type': 'multipart/form-data'
    }

    const url = 'http://localhost:5000/predict'

    var bodyFormData = new FormData();
    bodyFormData.set('text', text);

    const res: AxiosResponse = await axios({ url, method: 'post', headers, data: bodyFormData })
    if (res.status === 200) {
        if (res && res.data && res.data.done && res.data.score !== undefined) {
            // console.log('Res:', res);
            return res.data.score
        } else
            return null
    } else {
        console.error('Res error:', res);
        return null
    }
}

const App: React.FC = () => {

    const textRef = useRef<string | null>(null);
    const [estimatedScore, setEstimatedScore] = useState<number | null>(null)
    const [errorText, setErrorText] = useState(false)

    const classes = useStyles();

    const sumitEvaluation = (event: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
        event.preventDefault();
        setErrorText(textRef.current == null)
        if (textRef.current) {
            estimateReview(textRef.current).then((score) => {
                // console.log('Score:', score)
                setEstimatedScore(score)
            })
        }
    }

    let scoreStyle = classes.red
    switch (estimatedScore) {
        case (0):
            scoreStyle = classes.red
            break
        case (1):
            scoreStyle = classes.orange
            break
        case (2):
            scoreStyle = classes.yellow
            break
        case (3):
            scoreStyle = classes.blue
            break
        case (4):
            scoreStyle = classes.darkBlue
            break
        case (5):
            scoreStyle = classes.green
            break
        default:
            scoreStyle = classes.red
            break
    }

    return (
        <div className={classes.root}>
            <Grid item xs={12}>
                <Paper className={classes.paperHeader}>Food Review Estimator</Paper>
            </Grid>
            <Grid item xs={12}>
                <TextField className={classes.textField}
                    error={errorText}
                    margin="normal" type="string" defaultValue={textRef.current} onChange={e => textRef.current = e.target.value} id="standard-basic" label="Please enter food review text" />
            </Grid>
            <Grid item xs={2}>
                <Button variant="contained" color="primary" onClick={sumitEvaluation}>Estimate</Button>
            </Grid>
            {estimatedScore != null &&
                <Grid container spacing={3}>
                    <Grid item xs={3}>
                        <InputLabel id="estimatedRating">Estimated Rating</InputLabel>
                        <Paper className={`${scoreStyle} ${classes.paperOut}`}>{estimatedScore}</Paper>
                    </Grid>
                    <Grid item xs={9}>
                        <InputLabel id="reviewText" >Review Text</InputLabel>
                        <Paper className={classes.paperOut}>{textRef.current}</Paper>
                    </Grid>
                </Grid>}
        </div >
    );
}

export default App;
