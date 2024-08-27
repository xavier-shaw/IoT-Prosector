var mongoose = require('mongoose');
var Schema = mongoose.Schema;

const GOOGLE_HOME_INSTRUCTIONS = [
  {
    "function": "Turn off",
    "interaction": "Plug power cable",
    "image": "/plug.jpg"
  },
  {
    "function": "Play or pause",
    "interaction": "Tap",
    "image": "/tap.jpg"
  },
  {
    "function": "Start request",
    "interaction": "Say keyword",
    "image": "/press_hold.jpg"
  },
  {
    "function": "Do request",
    "interaction": "Say command",
    "image": ""
  },
  {
    "function": "Turn down volume",
    "interaction": "Swipe counter-clockwise",
    "image": "/volume_down.jpg"
  },
  {
    "function": "Turn up volume",
    "interaction": "Swipe clockwise",
    "image": "/volume_up.jpg"
  },
  {
    "function": "Mute or unmute",
    "interaction": "Press mic button",
    "image": "/press_mic.jpg"
  }]

var BoardSchema = new Schema({
  _id: {
    type: String,
  },
  title: {
    type: String,
    default: 'untitled'
  },
  created_date: {
    type: Date,
    default: Date.now
  },
  data: {
    type: Object,
    default: { 
      instructions: GOOGLE_HOME_INSTRUCTIONS,
      stateSequence: [],
      actionSequence: [] 
    },
  },
  chart: {
    type: Object,
    default: { nodes: [], edges: [] }
  }
});

module.exports = mongoose.model('boards', BoardSchema);