var SharedVariables = require("../models/sharedModel");

exports.start_sensing = function (req, res) {
    SharedVariables.findOneAndUpdate({ name: "sensing" }, { value: req.params.stage }, { new: true }).then((resp) => {
        if (resp) {
            res.json(resp)
        }
    })
        .catch((error) => {
            console.error("Error occurs:", error);
        });
};

exports.stop_sensing = function (req, res) {
    SharedVariables.findOneAndUpdate({ name: "sensing" }, { value: "false" }, { new: true }).then((resp) => {
        if (resp) {
            res.json(resp)
        }
    })
        .catch((error) => {
            console.error("Error occurs:", error);
        });
};

exports.update_device = function (req, res) {
    SharedVariables.findOneAndUpdate({ name: "device" }, { value: req.params.device }, { new: true }).then((resp) => {
        if (resp) {
            res.json(resp)
        }
    })
        .catch((error) => {
            console.error("Error occurs:", error);
        });
}