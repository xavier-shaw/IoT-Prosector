var Video = require("../models/videoModel");

exports.upload_video_by_id = function (req, res) {
    var new_video = new Video(req.body);
    new_video.save().then((video) => {
        res.json(video);
    })
        .catch((error) => {
            console.error("Error occurs:", error);
        }); 
};

exports.get_video_by_id = function (req, res) {
    Video.find({ node_id: req.params.id }).then((video) => {
        res.json(video);
    })
        .catch((error) => {
            console.error("Error occurs:", error);
        });
};