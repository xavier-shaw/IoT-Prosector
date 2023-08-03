var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var DataSchema = new Schema({
  _id: {
    type: String,
  },
  idx: {
    type: String,
  },
  state: {
    type: String,
  },
  time: {
    type: Number,
  },
  device: {
    type: String,
  },
  pos_x: {
    type: Number
  },
  pos_y: {
    type: Number
  },
  data: {
    type: Array,
  }
});

module.exports = mongoose.model('iotdatas', DataSchema);