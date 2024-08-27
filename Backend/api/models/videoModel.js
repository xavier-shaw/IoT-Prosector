var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var VideoSchema = new Schema({
  device: {
    type: String,
  },
  idx: {
    type: String,
  },
  video: {
    type: String
  }
});

module.exports = mongoose.model('actionVideos', VideoSchema);