CREATE TABLE `llm_survey` (
  `feedbackID` int(11) NOT NULL AUTO_INCREMENT,
  `sessionID` varchar(256) NOT NULL,
  `projectID` int(11) DEFAULT NULL,
  `rating` tinyint(1) NOT NULL,
  `feedbackIntent` enum('Encouragement','Review','Warning') NOT NULL,
  `predictedOutcome` varchar(32) DEFAULT NULL,
  `promptVersion` varchar(64) DEFAULT NULL,
  `modelVersion` varchar(64) DEFAULT NULL,
  `feedbackHash` char(64) DEFAULT NULL,
  `lastUpdate` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`feedbackID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;