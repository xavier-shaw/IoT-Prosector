var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var BoardSchema = new Schema({
  _id: {
    type: String,
  },
  title: {
    type: String,
    required: 'Kindly enter the title of the board'
  },
  created_date: {
    type: Date,
    default: Date.now
  },
  author: {
    type: String,
    default: "yuwei"
  },
  status: {
    type: String,
    // enum: ['pending', 'ongoing', 'completed']
    default: 'ongoing'
  },
  board_type: {
    type: String,
    // enum: ['eval', 'test', 'public', 'example'],
    default: 'example'
  },
  data: {
    type: String,
    default: '{}',
  },
  chart: {
    type: String,
    default: '{"offset":{"x":0,"y":0},"nodes":{},"links":{},"selected":{},"hovered":{}}'
  }
});

module.exports = mongoose.model('boards', BoardSchema);