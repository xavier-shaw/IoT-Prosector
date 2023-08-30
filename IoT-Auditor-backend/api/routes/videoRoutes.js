module.exports = function (app) {
    var video = require('../controllers/videoController');

    app.route('/api/video/get/:id')
        .get(video.get_video_by_id);
    
    app.route('/api/video/upload')
        .post(video.upload_video_by_id)
};

