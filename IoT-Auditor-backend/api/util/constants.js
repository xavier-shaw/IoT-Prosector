module.exports = Object.freeze({
    QUALTRIC_CHOICE_MAPPING : {1: "Extremely comfortable",
      2: "Somewhat comfortable",
      3: "Neither comfortable or uncomfortable",
      4: "Somewhat uncomfortable",
      5: "Extremely uncomfortable"},
    QUALTRIC_INVERSE_MAPPING : {"Extremely comfortable": 1,
      "Somewhat comfortable": 2,
      "Neither comfortable or uncomfortable": 3,
      "Somewhat uncomfortable": 4,
      "Extremely uncomfortable": 5},
    CONCERN_FIELDS_MAP : {
      "Answer.cb_algorithm.on": "Lack of trust for algorithms",
      "Answer.cb_alternative.on": "Lack of alternative choice",
      "Answer.cb_anonymization.on": "Insufficient anonymization",
      "Answer.cb_autonomy.on": "Lack of respect for autonomy",
      "Answer.cb_bias.on": "Bias or discrimination",
      "Answer.cb_datainsecurity.on": "Insufficient data security",
      "Answer.cb_deception.on": "Deceptive data practice",
      "Answer.cb_informedconsent.on": "Lack of informed consent",
      "Answer.cb_invasive.on": "Invasive monitoring",
      "Answer.cb_mutualbenefits.on": "Data commodification",
      "Answer.cb_nocontrol.on": "Lack of control of personal data",
      "Answer.cb_toohighrisks.on": "Too high potential risks",
      // "Answer.cb_toolowbenefits.on": "Too low potential benefits",
      "Answer.cb_unexpected.on": "Violation of expectations",
      // "Answer.cb_violationofconsent.on": "Violation of consent",
      "Answer.cb_vulnerable.on": "Lack of protection for vulnerable populations",
      "Answer.other_concerns": "Other privacy concerns",
      "Answer.cb_badresponse.on": "Bad response",
      "Answer.cb_nonconcern.on": "No privacy concern"
    }
  });