var PredictState = require("../models/predictModel");

exports.get_predict_state = function (req, res) {
    PredictState.find({ device: req.params.device }).then((state) => {
        res.json(state);
    })
        .catch((error) => {
            console.error("Error occurs:", error);
        });
};